/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include "Common.metal"
#include <metal_stdlib>
using namespace metal;

kernel void pixel_unshuffle(texture2d_array<ftype, access::sample> inTexture[[texture(0)]],
    texture2d_array<ftype, access::write> outTexture[[texture(1)]],
    constant PixelUnShuffleParam& param[[buffer(0)]],
    uint3 gid[[thread_position_in_grid]]) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
        gid.z >= outTexture.get_array_size())
        return;

    int downscale_factor = param.downscale_factor;
    int outX = gid.x * downscale_factor;
    int outY = gid.y * downscale_factor;

    ftype4 res = ftype4(0.0);

    for (int i = 0; i < 4; i++) {
        int c = gid.z * 4 + i;
        int outC = c / (downscale_factor * downscale_factor);
        int offset = c % (downscale_factor * downscale_factor);
        int offset_h = offset / downscale_factor;
        int offset_w = offset % downscale_factor;

        int readX = outX + offset_w;
        int readY = outY + offset_h;

        ftype4 input = inTexture.read(uint2(readX, readY), outC / 4);
        res[i] = input[outC % 4];
    }
    outTexture.write(res, gid.xy, gid.z);
}
