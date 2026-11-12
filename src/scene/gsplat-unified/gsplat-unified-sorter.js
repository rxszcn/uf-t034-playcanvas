import { EventHandler } from '../../core/event-handler.js';
import { platform } from '../../core/platform.js';
import { UnifiedSortWorker } from './gsplat-unified-sort-worker.js';
import { GSplatSortBinWeights } from './gsplat-sort-bin-weights.js';

/**
 * @import { GSplatInfo } from './gsplat-info.js'
 * @import { Scene } from '../scene.js'
 */

/** @type {Set<number>} */
const _neededIds = new Set();

// Camera inputs below this delta are treated as unchanged. The sort keys quantize depth to 16-20
// bits, so sub-epsilon camera motion cannot change any key anyway, and the stable tie-break order of
// equal-key splats must not be re-rolled just because the per-frame matrix rebuild drifted a ulp.
const SORT_INPUT_EPSILON = 1e-6;

class GSplatUnifiedSorter extends EventHandler {
    worker;

    bufferLength = 0;

    availableOrderData = [];

    // track how many jobs are in flight
    jobsInFlight = 0;

    // true if we have new version to process
    hasNewVersion = false;

    /**
     * Camera inputs of the last dispatched sort, used to skip a redundant re-sort of a still scene.
     * Null after a world-state change ({@link GSplatUnifiedSorter#setSortParameters}) or centers
     * change, which always require a fresh sort.
     *
     * @type {{radialSorting: boolean, nearClip: number, logDepthKeys: boolean, params: object[]}|null}
     * @private
     */
    _lastSortInputs = null;

    /**
     * Pending sorted result to be applied next frame. If multiple sorted results are received from
     * the worker, the latest result is stored here.
     *
     * @type {{ count: number, version: number, orderData: Uint32Array }|null}
     */
    pendingSorted = null;

    /** @type {Set<number>} */
    centersSet = new Set();

    /** @type {boolean} */
    _destroyed = false;

    /** @type {Scene|null} */
    scene = null;

    /**
     * @param {Scene} [scene] - The scene to fire sort timing events on.
     */
    constructor(scene) {
        super();
        this.scene = scene ?? null;

        // Build worker source with GSplatSortBinWeights class injected via stringification.
        const workerPort = platform.environment === 'node' ? 'require("node:worker_threads").parentPort' : 'self';
        const workerSource = `
            const GSplatSortBinWeights = ${GSplatSortBinWeights.toString()};
            (${UnifiedSortWorker.toString()})(${workerPort})
        `;

        if (platform.environment === 'node') {
            this.worker = new Worker(workerSource, {
                eval: true
            });
            this.worker.on('message', this.onSorted.bind(this));
        } else {
            this.worker = new Worker(URL.createObjectURL(new Blob([workerSource], {
                type: 'application/javascript'
            })));
            this.worker.addEventListener('message', this.onSorted.bind(this));
        }
    }

    onSorted(message) {

        if (this._destroyed) {
            return;
        }

        const msgData = message.data ?? message;

        // Fire sortTime event directly on scene (before result might be dropped)
        if (this.scene && msgData.sortTime !== undefined) {
            this.scene.fire('gsplat:sorted', msgData.sortTime);
        }

        const orderData = new Uint32Array(msgData.order);

        // decrement jobs in flight counter
        this.jobsInFlight--;

        // if there's already a pending result, return its orderData to the pool
        if (this.pendingSorted) {
            this.releaseOrderData(this.pendingSorted.orderData);
        }

        // store the result to be available
        this.pendingSorted = {
            count: msgData.count,
            version: msgData.version,
            orderData: orderData
        };
    }

    applyPendingSorted() {
        if (this.pendingSorted) {
            const { count, version, orderData } = this.pendingSorted;
            this.pendingSorted = null;
            this.fire('sorted', count, version, orderData);

            // reuse order data
            this.releaseOrderData(orderData);
        }
    }

    releaseOrderData(orderData) {
        if (orderData.length === this.bufferLength) {
            this.availableOrderData.push(orderData);
        }
    }

    destroy() {
        this._destroyed = true;
        this.pendingSorted = null;
        this.worker.terminate();
        this.worker = null;
    }

    /**
     * Adds or removes centers from the sorter.
     *
     * @param {number} id - The id of the centers.
     * @param {Float32Array|null} centers - The centers buffer.
     */
    setCenters(id, centers) {

        if (centers) { // add

            if (!this.centersSet.has(id)) {
                this.centersSet.add(id);

                // the centers feed the distance keys, so a changed set invalidates the still-camera
                // cache and forces the next sort
                this._lastSortInputs = null;

                // clone centers buffer - required when multiple workers sort the same splat resource
                const centersBuffer = centers.buffer.slice();

                // post centers to worker
                this.worker.postMessage({
                    command: 'addCenters',
                    id: id,
                    centers: centersBuffer
                }, [centersBuffer]);
            }

        } else { // remove

            if (this.centersSet.has(id)) {
                this.centersSet.delete(id);

                this._lastSortInputs = null;

                // post centers to worker
                this.worker.postMessage({
                    command: 'removeCenters',
                    id: id
                });
            }
        }
    }

    /**
     * Updates centers in the worker based on current splats.
     * Adds new centers and removes centers no longer needed.
     *
     * @param {GSplatInfo[]} splats - Array of active splat infos.
     */
    updateCentersForSplats(splats) {

        // collect resource IDs from current splats
        for (const splat of splats) {
            const id = splat.resource.id;
            _neededIds.add(id);

            // add centers if not already present
            if (!this.centersSet.has(id)) {
                this.setCenters(id, splat.resource.centers);
            }
        }

        // remove centers no longer needed
        for (const id of this.centersSet) {
            if (!_neededIds.has(id)) {
                this.setCenters(id, null);
            }
        }

        _neededIds.clear();
    }

    /**
     * Sets sort parameters data for sorting of splats.
     *
     * @param {object} payload - The sort parameters payload to send.
     */
    setSortParameters(payload) {

        // we have a new version to process
        this.hasNewVersion = true;

        // a new world state changes the index map and active splat set, so any cached still-camera
        // result no longer applies
        this._lastSortInputs = null;

        // output size matches input size, clear available data when it changes
        const { textureSize } = payload;
        const newLength = textureSize * textureSize;
        if (newLength !== this.bufferLength) {
            this.bufferLength = newLength;
            this.availableOrderData.length = 0;
        }

        this.worker.postMessage(payload);
    }

    /**
     * Compares a sort request's camera inputs against those of the last dispatched sort. Returns
     * true only when the sort mode and every per-placement direction, position and scalar are within
     * {@link SORT_INPUT_EPSILON}; such a request would produce a bit-identical order, so re-sorting
     * only risks re-rolling the stable tie-break of equal-key splats.
     *
     * @param {object[]} params - The per-placement sort parameters.
     * @param {boolean} radialSorting - Whether radial distance sorting is used.
     * @param {number} nearClip - The camera's near clip distance.
     * @param {boolean} logDepthKeys - Whether directional keys are log spaced.
     * @returns {boolean} True when the inputs are effectively unchanged.
     * @private
     */
    _inputsUnchanged(params, radialSorting, nearClip, logDepthKeys) {
        const prev = this._lastSortInputs ?? null;
        if (!prev ||
            prev.radialSorting !== radialSorting ||
            prev.logDepthKeys !== logDepthKeys ||
            prev.nearClip !== nearClip ||
            prev.params.length !== params.length) {
            return false;
        }

        for (let i = 0; i < params.length; i++) {
            const a = params[i];
            const b = prev.params[i];
            const ad = a.transformedDirection;
            const bd = b.transformedDirection;
            const ap = a.transformedPosition;
            const bp = b.transformedPosition;
            if (Math.abs(ad.x - bd.x) > SORT_INPUT_EPSILON ||
                Math.abs(ad.y - bd.y) > SORT_INPUT_EPSILON ||
                Math.abs(ad.z - bd.z) > SORT_INPUT_EPSILON ||
                Math.abs(ap.x - bp.x) > SORT_INPUT_EPSILON ||
                Math.abs(ap.y - bp.y) > SORT_INPUT_EPSILON ||
                Math.abs(ap.z - bp.z) > SORT_INPUT_EPSILON ||
                Math.abs(a.offset - b.offset) > SORT_INPUT_EPSILON ||
                Math.abs(a.scale - b.scale) > SORT_INPUT_EPSILON) {
                return false;
            }
        }

        return true;
    }

    /**
     * Caches a compact copy of the camera inputs a sort was dispatched with, for the next
     * {@link GSplatUnifiedSorter#_inputsUnchanged}. AABBs and intervals change through
     * {@link GSplatUnifiedSorter#setSortParameters} and invalidate the cache instead.
     *
     * @param {object[]} params - The per-placement sort parameters.
     * @param {boolean} radialSorting - Whether radial distance sorting is used.
     * @param {number} nearClip - The camera's near clip distance.
     * @param {boolean} logDepthKeys - Whether directional keys are log spaced.
     * @private
     */
    _cacheInputs(params, radialSorting, nearClip, logDepthKeys) {
        this._lastSortInputs = {
            radialSorting,
            nearClip,
            logDepthKeys,
            params: params.map(p => ({
                transformedDirection: { ...p.transformedDirection },
                transformedPosition: { ...p.transformedPosition },
                offset: p.offset,
                scale: p.scale
            }))
        };
    }

    /**
     * Sends sorting parameters to the sorter. Called every frame sorting is needed.
     *
     * @param {object} params - The sorting parameters - per-splat directions, offsets, scales, AABBs.
     * @param {boolean} radialSorting - Whether to use radial distance sorting.
     * @param {number} nearClip - The camera's near clip distance, which log spaced sort keys start
     * inside of.
     * @param {boolean} logDepthKeys - Whether a directional sort's keys are log spaced, which needs
     * a perspective, non-fisheye camera (radial sort keys always are).
     * @returns {boolean} True if a sort job was sent; false if one is in flight for this version,
     * in which case the caller asks again later, or the camera was still and the previous order
     * stays in effect.
     */
    setSortParams(params, radialSorting, nearClip, logDepthKeys) {

        // Back-pressure takes priority: while a job of this version is in flight, turn the request
        // away unchanged so the caller asks again when the worker is free.
        if (!this.hasNewVersion && this.jobsInFlight > 0) {
            return false;
        }

        // A still scene re-sorts to a bit-identical order. Skip it rather than re-rolling the stable
        // tie-break of equal-key splats (which shows up as on-demand frame jumps), and report the
        // request as handled so the manager clears its sort flag. A new world state or centers change
        // clears the cache and forces the sort.
        if (!this.hasNewVersion && this._inputsUnchanged(params, radialSorting, nearClip, logDepthKeys)) {
            return true;
        }

        // reuse or allocate new order data
        let orderData = this.availableOrderData.pop();
        if (!orderData) {
            orderData = new Uint32Array(this.bufferLength);
        }

        // worker management
        this.jobsInFlight++;
        this.hasNewVersion = false;

        this._cacheInputs(params, radialSorting, nearClip, logDepthKeys);

        // send job to worker
        this.worker.postMessage({
            command: 'sort',
            sortParams: params,
            radialSorting: radialSorting,
            nearClip: nearClip,
            logDepthKeys: logDepthKeys,
            order: orderData.buffer
        }, [
            orderData.buffer
        ]);
        return true;
    }
}

export { GSplatUnifiedSorter };
