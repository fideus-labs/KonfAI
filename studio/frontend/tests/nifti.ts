// SPDX-License-Identifier: Apache-2.0
//
// The smallest volume the viewer reads: an 8x8x8 uint8 NIfTI-1, gzipped, built here so no test
// depends on a file and none carries patient data.
import { gzipSync } from "node:zlib";

export function tinyNifti(): Buffer {
  const header = Buffer.alloc(352);
  header.writeInt32LE(348, 0);
  const dims = [3, 8, 8, 8, 1, 1, 1, 1];
  dims.forEach((d, i) => header.writeInt16LE(d, 40 + 2 * i));
  header.writeInt16LE(2, 70); // datatype uint8
  header.writeInt16LE(8, 72); // bitpix
  [0, 1, 1, 1, 1, 1, 1, 1].forEach((p, i) => header.writeFloatLE(p, 76 + 4 * i));
  header.writeFloatLE(352, 108); // vox_offset
  header.writeFloatLE(1, 112); // scl_slope
  header.writeInt16LE(0, 252); // qform_code
  header.writeInt16LE(1, 254); // sform_code
  [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
  ].forEach((row, r) => row.forEach((v, c) => header.writeFloatLE(v, 280 + 16 * r + 4 * c)));
  header.write("n+1\0", 344, "ascii");
  const voxels = Buffer.alloc(8 * 8 * 8);
  for (let i = 0; i < voxels.length; i++) voxels[i] = i % 256;
  return gzipSync(Buffer.concat([header, voxels]));
}
