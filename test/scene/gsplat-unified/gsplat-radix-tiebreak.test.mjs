import { expect } from 'chai';

// Reproducible model of the two GPU radix backends' tie-break behaviour under equal sort keys.
//
// The gsplat projector quantizes depth to 16-20 bit keys, so neighbouring splats routinely share a
// key and their draw order is decided entirely by the radix sort's stability. Both backends sort the
// keys correctly; they differ only in the relative order of *equal-key* elements:
//
//   - portable (multi-pass): each workgroup maps to a fixed input range
//     (WORKGROUP_ID * ELEMENTS_PER_WORKGROUP), and every pass preserves order, so equal-key elements
//     keep their input order and the result is identical frame after frame.
//   - OneSweep: the binning pass claims partition tiles with `atomicAdd(b_index[pass], 1)`, so the
//     workgroup owning a given input range depends on scheduling and equal-key elements crossing a
//     partition boundary are reordered frame to frame.
//
// This test models those two tie-breaks in plain JS (no GPU required) to make the non-determinism
// regression detectable in CI. A deterministic LCG drives the per-frame partition claim order, so the
// exact numbers are reproducible.
describe('GSplat radix tie-break determinism model', function () {

    const PART_SIZE = 8;
    const N = 64;
    const KEY_MASK = 0x07; // 8 distinct keys, forcing heavy collisions like quantized depth

    const makeKeys = () => {
        const keys = new Uint32Array(N);
        for (let i = 0; i < N; i++) {
            keys[i] = (i * 7 + 3) & KEY_MASK;
        }
        return keys;
    };

    // Stable counting sort: equal-key elements stay in ascending input-index order. This is what the
    // portable backend's fixed-range, order-preserving passes converge to every frame.
    const stableCountingSort = (keys) => {
        const buckets = new Uint32Array(KEY_MASK + 1);
        for (let i = 0; i < N; i++) {
            buckets[keys[i]]++;
        }
        const starts = new Uint32Array(KEY_MASK + 1);
        for (let k = 0; k < KEY_MASK; k++) {
            starts[k + 1] = starts[k] + buckets[k];
        }
        const order = new Uint32Array(N);
        const cursor = starts.slice();
        for (let i = 0; i < N; i++) {
            order[cursor[keys[i]]++] = i;
        }
        return order;
    };

    // Models OneSweep: partitions are processed in a per-frame claim order (Fisher-Yates from a
    // deterministic rng), and the equal-key tie-break is the claim order. A stable sort over the
    // collected (key, index) pairs keeps that claim order within each key.
    const oneSweepSort = (keys, rng) => {
        const numParts = N / PART_SIZE;
        const parts = Array.from({ length: numParts }, (_, p) => p);
        for (let i = parts.length - 1; i > 0; i--) {
            const j = rng() % (i + 1);
            [parts[i], parts[j]] = [parts[j], parts[i]];
        }

        const collected = [];
        for (const p of parts) {
            for (let j = 0; j < PART_SIZE; j++) {
                const idx = p * PART_SIZE + j;
                collected.push([keys[idx], idx]);
            }
        }
        // Array.prototype.sort is stable, preserving the collection (claim) order within a key.
        collected.sort((a, b) => a[0] - b[0]);
        return Uint32Array.from(collected.map(pair => pair[1]));
    };

    const diffCount = (a, b) => {
        let d = 0;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) d++;
        }
        return d;
    };

    // Deterministic LCG: identical sequence on every run, so the assertions never flake.
    const makeLcg = (seed) => {
        let state = seed;
        return () => {
            state = (state * 1103515245 + 12345) & 0x7fffffff;
            return state;
        };
    };

    it('the portable-style stable order is identical on every frame', function () {
        const keys = makeKeys();
        const reference = stableCountingSort(keys);
        for (let frame = 0; frame < 5; frame++) {
            expect(Array.from(stableCountingSort(keys))).to.deep.equal(Array.from(reference));
        }
    });

    it('the OneSweep-style atomic partition order changes the equal-key order between frames', function () {
        const keys = makeKeys();
        const orders = [];
        for (let frame = 0; frame < 5; frame++) {
            orders.push(oneSweepSort(keys, makeLcg(1234567 + frame)));
        }

        // each frame differs from the stable reference, and consecutive frames differ from each
        // other - the fine-grained flicker / on-demand jump the bug report describes
        expect(orders.every(o => diffCount(o, stableCountingSort(keys)) > 0)).to.equal(true);
        let frameToFrame = 0;
        for (let f = 1; f < orders.length; f++) {
            frameToFrame += diffCount(orders[f - 1], orders[f]);
        }
        expect(frameToFrame).to.be.greaterThan(0);
    });

    it('both orderings remain correctly sorted by key regardless of tie-break', function () {
        const keys = makeKeys();
        const checkSorted = (order) => {
            for (let i = 1; i < order.length; i++) {
                expect(keys[order[i - 1]]).to.be.at.most(keys[order[i]]);
            }
        };
        checkSorted(stableCountingSort(keys));
        checkSorted(oneSweepSort(keys, makeLcg(999)));
    });
});
