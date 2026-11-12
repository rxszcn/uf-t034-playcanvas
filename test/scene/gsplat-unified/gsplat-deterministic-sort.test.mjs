import { expect } from 'chai';

import { NullGraphicsDevice } from '../../../src/platform/graphics/null/null-graphics-device.js';
import { ComputeRadixSort } from '../../../src/scene/graphics/radix-sort/compute-radix-sort.js';
import { GSplatHybridRenderer } from '../../../src/scene/gsplat-unified/gsplat-hybrid-renderer.js';

// Determinism of the gsplat order. The depth keys are quantized to 16-20 bits, so neighbouring
// splats routinely share a key and their draw order is the radix sort's stable tie-break. The
// hybrid renderer must therefore use a genuinely stable backend even on hardware where AUTO would
// pick the non-deterministic OneSweep (whose binning pass claims partition tiles with an atomic
// counter, reordering equal-key splats frame to frame).
describe('GSplat deterministic sorting', function () {

    // The Null device has no compute backend; stub what the sorters need to build pipelines and
    // allocate buffers (dispatches are no-ops). Mirrors the radix-sort allocation test harness.
    const createNvidiaDevice = () => {
        const device = new NullGraphicsDevice({ width: 16, height: 16 });
        device.supportsCompute = true;
        device.supportsSubgroups = true;
        device.minSubgroupSize = 32;
        device.maxSubgroupSize = 32;
        // NVIDIA is the OneSweep target: without the renderer pinning the backend, AUTO selects the
        // non-deterministic atomic-partition sorter here.
        device.gpuAdapter = { info: { vendor: 'nvidia' } };
        device.limits = { maxComputeWorkgroupsPerDimension: 65535 };
        device.createBufferImpl = () => ({ allocate() {}, destroy() {}, clear() {}, loseContext() {}, restoreContext() {} });
        device.createComputeImpl = () => ({ destroy() {} });
        return device;
    };

    describe('hybrid GPU renderer backend selection', function () {

        it('AUTO picks OneSweep on the NVIDIA device, proving the pin is load-bearing', function () {
            const device = createNvidiaDevice();
            const auto = new ComputeRadixSort(device, { indirect: true });
            // OneSweep uses 8-bit radix passes, the portable backend 4-bit.
            expect(auto.radixBits).to.equal(8);
            auto.destroy();
            device.destroy();
        });

        it('pins the gsplat sorter to the stable portable backend on an NVIDIA device', function () {
            const device = createNvidiaDevice();
            const renderer = Object.create(GSplatHybridRenderer.prototype);
            renderer.device = device;
            renderer.gpuSorter = null;
            // the projector and compaction stay stubbed so only the sorter backend is under test
            renderer.projector = {};
            renderer.intervalCompaction = {};

            GSplatHybridRenderer.prototype._ensureGpuPipeline.call(renderer, true);

            expect(renderer.gpuSorter).to.not.equal(null);
            // 4-bit radix identifies the portable backend, whose workgroups map to fixed input
            // ranges and therefore keep equal-key splats in a stable, frame-independent order.
            expect(renderer.gpuSorter.radixBits).to.equal(4);

            renderer.gpuSorter.destroy();
            device.destroy();
        });

        it('keeps the portable backend across repeated pipeline requests', function () {
            const device = createNvidiaDevice();
            const renderer = Object.create(GSplatHybridRenderer.prototype);
            renderer.device = device;
            renderer.gpuSorter = null;
            renderer.projector = {};
            renderer.intervalCompaction = {};

            GSplatHybridRenderer.prototype._ensureGpuPipeline.call(renderer, true);
            const first = renderer.gpuSorter;
            // subsequent calls must retain the same stable sorter rather than re-selecting AUTO
            GSplatHybridRenderer.prototype._ensureGpuPipeline.call(renderer, true);
            expect(renderer.gpuSorter).to.equal(first);
            expect(renderer.gpuSorter.radixBits).to.equal(4);

            first.destroy();
            device.destroy();
        });
    });
});
