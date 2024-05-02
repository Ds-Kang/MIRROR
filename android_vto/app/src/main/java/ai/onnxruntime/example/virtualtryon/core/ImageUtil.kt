/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.onnxruntime.example.virtualtryon.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color.argb
import android.util.Log
import java.io.IOException
import java.lang.Math.max
import java.lang.Math.min


const val DIM_BATCH_SIZE = 1
const val DIM_PIXEL_SIZE = 3
const val IMAGE_SIZE_X = 540
const val IMAGE_SIZE_Y = 960


// extension function to get bitmap from assets
fun Context.assetsToBitmap(fileName: String): Bitmap? {
    return try {
        with(assets.open(fileName)) {
            BitmapFactory.decodeStream(this)
        }
    } catch (e: IOException) {
        null
    }
}

fun preProcess(bitmap: Bitmap, channel: Int, height: Int, width: Int): ByteArray {
    val imgData = ByteArray(channel * height * width)
    val stride = width * height
    val bmpData = IntArray(stride)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    for (j in 0 until height) {
        for (i in 0 until width) {
            val idx = width * j + i
            val pixelValue = bmpData[idx]
            imgData[idx * 3] = (pixelValue shr 16 and 0xFF).toByte()
            imgData[idx * 3 + 1] = (pixelValue shr 8 and 0xFF).toByte()
            imgData[idx * 3 + 2] = (pixelValue and 0xFF).toByte()
        }
    }

    return imgData
}

fun preProcessCloth(bitmap: Bitmap, channel: Int, height: Int, width: Int): ByteArray {
    val imgData = ByteArray(channel * height * width)
    val stride = bitmap.width * bitmap.height
    val bmpData = IntArray(stride)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    for (j in 0 until height) {
        for (i in 0 until width) {
            val idx = width * j + i
            val pixelValue = bmpData[idx]
            imgData[idx * 3] = (pixelValue shr 16 and 0xFF).toByte()
            imgData[idx * 3 + 1] = (pixelValue shr 8 and 0xFF).toByte()
            imgData[idx * 3 + 2] = (pixelValue and 0xFF).toByte()
        }
    }

    Log.d("ImageUtil", "preProcess: ${imgData.size}")
    return imgData
}

fun preProcessGray(bitmap: Bitmap, height: Int, width: Int): ByteArray {
    val imgData = ByteArray(height * width)
    val bmpData = IntArray(width * height)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

    for (j in 0 until height) {
        for (i in 0 until width) {
            val idx = width * j + i
            val pixelValue = bmpData[idx]

            imgData[idx] = (pixelValue and 0xFF).toByte()
        }
    }


    return imgData
}

fun clipFloatToInt(fValue: Float): Int {
    var iValue: Int = ((fValue + 1.0) * 0.5 * 255).toInt()
    if (iValue > 255) {
        iValue = 255
    } else if (iValue < 0) {
        iValue = 0
    }
    return iValue
}

fun moveBB(firstBB: IntArray, diff: IntArray): IntArray {
    var movedBB = IntArray(8)

    movedBB[0] = max(firstBB[0] + diff[0], 0)
    movedBB[1] = min(firstBB[1] + diff[0], 540)
    movedBB[2] = max(firstBB[2] + diff[1], 0)
    movedBB[3] = min(firstBB[3] + diff[1], 960)
    movedBB[4] = max(firstBB[4] + diff[0], 0)
    movedBB[5] = min(firstBB[5] + diff[0], 540)
    movedBB[6] = max(firstBB[6] + diff[1], 0)
    movedBB[7] = min(firstBB[7] + diff[1], 960)

    return movedBB
}

fun moveP0(firstP0: FloatArray, moveLen: Float): FloatArray {
    var movedP0 = FloatArray(firstP0.size)

    for (i in firstP0.indices) {
        movedP0[i] = firstP0[i] + moveLen
    }
    return movedP0
}


fun toBitmap(floatArray: FloatArray): Bitmap {
    val width = 192
    val height = 256
    val channel = 3

    val intValues = IntArray(width * height)
    for (k in 0 until height) {
        for (j in 0 until width) {
            var r = floatArray.get(0 * width * height + k * width + j)
            var g = floatArray.get(1 * width * height + k * width + j)
            var b = floatArray.get(2 * width * height + k * width + j)
            var color = argb(255, clipFloatToInt(r), clipFloatToInt(g), clipFloatToInt(b))

            intValues[k * width + j] = color
        }
    }

    val conf = Bitmap.Config.ARGB_8888 // see other conf types

    val outputBitmap = Bitmap.createBitmap(width, height, conf) // this creates a MUTABLE bitmap
    outputBitmap.setPixels(
        intValues, 0, outputBitmap.width, 0, 0,
        outputBitmap.width, outputBitmap.height
    )

    return outputBitmap
}


fun toBitmap(byteArray: ByteArray, channel: Int, height: Int, width: Int): Bitmap {
    val intValues = IntArray(width * height)
    for (h in 0 until height) {
        for (w in 0 until width) {
            var r = byteArray[h * width * channel + w * channel + 0].toInt()
            var g = byteArray[h * width * channel + w * channel + 1].toInt()
            var b = byteArray[h * width * channel + w * channel + 2].toInt()

            if (r < 0) r += 128
            if (g < 0) g += 128
            if (b < 0) b += 128

            var color = argb(255, r, g, b)

            intValues[h * width + w] = color
        }
    }

    val conf = Bitmap.Config.ARGB_8888 // see other conf types

    val outputBitmap = Bitmap.createBitmap(width, height, conf) // this creates a MUTABLE bitmap
//    val outputBitmap = bmpSegmentation.copy(bmpSegmentation.config, true)
    outputBitmap.setPixels(
        intValues, 0, outputBitmap.width, 0, 0,
        outputBitmap.width, outputBitmap.height
    )

    return outputBitmap
}

fun toByteimage(intArray: IntArray, channel: Int, height: Int, width: Int): ByteArray {
    val outputArray = ByteArray(3 * width * height)
    for (h in 0 until height) {
        for (w in 0 until width) {
            for (c in 0 until 3) {
                outputArray[h * width * channel + w * channel + c] =
                    intArray.get(h * width * channel + w * channel + 2 - c).toByte()
            }
        }
    }

    return outputArray
}

fun toRGB(intArray: ByteArray, channel: Int, height: Int, width: Int): ByteArray {
    val outputArray = ByteArray(3 * width * height)
    for (h in 0 until height) {
        for (w in 0 until width) {
            for (c in 0 until 3) {
                outputArray[h * width * channel + w * channel + c] =
                    intArray.get(h * width * channel + w * channel + 2 - c)
            }
        }
    }

    return outputArray
}

fun getRatio(bb: IntArray, height: Float, width: Float): Float {
    val old_w = (bb[1] - bb[0]).toFloat()
    val old_h = (bb[3] - bb[2]).toFloat()
    if ((old_h / old_w) > (height / width)) {
        return height / old_h
    } else {
        return width / old_w
    }
}

fun getResizeSize(bb: IntArray, ratio: Float): IntArray {
    val oldWidth = (bb[1] - bb[0]).toFloat()
    val oldHeight = (bb[3] - bb[2]).toFloat()
    return intArrayOf((oldWidth * ratio).toInt(), (oldHeight * ratio).toInt())
}

fun getPaddings(new_size: IntArray, height: Int, width: Int): IntArray {
    val deltaWidth = width - new_size[0]
    val deltaHeight = height - new_size[1]

    return intArrayOf(0, deltaHeight, deltaWidth / 2, deltaWidth - (deltaWidth) / 2)
}

fun getCorrSize(bb: IntArray, ratio: Float): IntArray {
    return intArrayOf(
        ((bb[4] - bb[0]).toFloat() * ratio).toInt(),
        ((bb[5] - bb[0]).toFloat() * ratio).toInt(),
        ((bb[6] - bb[2]).toFloat() * ratio).toInt(),
        ((bb[7] - bb[2]).toFloat() * ratio).toInt()
    )
}


fun getP1(p0WithFirst: FloatArray, size: Int): FloatArray {
    val result = FloatArray(size)
    for (i in 0 until size) {
        result[i] = p0WithFirst[i]
    }
    return result
}

fun getFirstP0(p0WithFirst: FloatArray, size: Int): FloatArray {
    val result = FloatArray(size)
    for (i in 0 until size) {
        result[i] = p0WithFirst[i + size]
    }
    return result

}

fun getDiff(p1: FloatArray, firstP1: FloatArray): IntArray {
    var diffX = 0.0F
    var diffY = 0.0F
    val len = p1.size
    for (i in 0 until len step (2)) {
        diffX += (p1[i] - firstP1[i])
        diffY += (p1[i + 1] - firstP1[i + 1])
    }
    return intArrayOf((diffX / len.toFloat()).toInt(), (diffY / len.toFloat()).toInt())
}
