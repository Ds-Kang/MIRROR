package ai.onnxruntime.example.virtualtryon.core

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import java.io.*


internal fun readIndex(videoNumber: String): BooleanArray? {
    val dirPath = FileSource.PREVIEW_RESULT_PATH + "/" + videoNumber
    val directory = File(dirPath)
    if (!directory.exists()) {
        directory.mkdirs()
    }
    val filePath = "$dirPath/indexArray"
    val file = File(filePath)
    if (file.exists()) {
        try {
            val fis = FileInputStream(file)
            val ois = ObjectInputStream(fis)
            val returnlist = ois.readObject() as BooleanArray
            ois.close()
            Log.i("Load Index: ", filePath)
            return returnlist
        } catch (e: java.lang.Exception) {
            return null
        }
    } else {
        return null
    }
}

internal fun saveIndex(videoNumber: String, dnnIndex: BooleanArray) {
    val dirPath = FileSource.PREVIEW_RESULT_PATH + "/" + videoNumber
    val directory = File(dirPath)
    if (!directory.exists()) {
        directory.mkdirs()
    }

    val filePath = "$dirPath/indexArray"
    var out: ObjectOutput? = null
    try {
        out = ObjectOutputStream(FileOutputStream(File(filePath)))
        out.writeObject(dnnIndex)
        out.close()
    } catch (e: java.lang.Exception) {
        e.printStackTrace()
    }
}

internal fun readDnnInfo(videoNumber: String, frameNumber: String): VTO.DNNInfo? {
    val dirPath = FileSource.DNN_INFO_PATH + "/" + videoNumber
    val directory = File(dirPath)
    if (!directory.exists()) {
        directory.mkdirs()
    }

    val filePath = "$dirPath/$frameNumber" + "dnnInfo"

    val file = File(filePath)
    if (file.exists()) {
        try {
            val fis = FileInputStream(file)
            val ois = ObjectInputStream(fis)
            val returnlist = ois.readObject() as VTO.DNNInfo
            ois.close()
            Log.i("Load DNN Info: ", filePath)
            return returnlist
        } catch (e: java.lang.Exception) {
            val sw = StringWriter()
            val pw = PrintWriter(sw)
            e.printStackTrace(pw)
            val stackTraceString = sw.toString()

            // Log the complete stack trace
            Log.e("Read DNN Info Error", stackTraceString)
            return null
        }
    } else {
        return null
    }
}

internal fun writeDnnInfo(
    dnn_info: VTO.DNNInfo,
    videoNumber: String,
    frameNumber: String
) {
    val dirPath = FileSource.DNN_INFO_PATH + "/" + videoNumber
    val directory = File(dirPath)
    if (!directory.exists()) {
        directory.mkdirs()
    }
    val filePath = "$dirPath/$frameNumber" + "dnnInfo"

    var out: ObjectOutput? = null
    try {
        out = ObjectOutputStream(FileOutputStream(File(filePath)))
        out.writeObject(dnn_info)
        out.close()
    } catch (e: java.lang.Exception) {
        Log.e("FileUtils", "writeDnnInfo: ${e.stackTrace}")
    }
}

internal fun readLitInfo(
    videoNumber: String,
    frameNumber: String,
    limited: Boolean
): VTO.LITInfo? {
    val dirPath = FileSource.LIT_INFO_PATH + "/" + videoNumber
    val directory = File(dirPath)
    if (!directory.exists()) {
        directory.mkdirs()
    }
    val filePath = "$dirPath/$frameNumber" + "litInfo"

    val file = File(filePath)
    if (file.exists() && !limited) {
        try {
            val fis = FileInputStream(file)
            val ois = ObjectInputStream(fis)
            val returnlist = ois.readObject() as VTO.LITInfo
            ois.close()
            // Log.i("Load LIT Info: ", filePath)
            return returnlist
        } catch (e: java.lang.Exception) {
            e.printStackTrace()
            return null
        }
    } else {
        return null
    }
}

internal fun writeLitInfo(
    lit_info: VTO.LITInfo,
    videoNumber: String,
    frameNumber: String
) {
    val dirPath = FileSource.LIT_INFO_PATH + "/" + videoNumber
    val directory = File(dirPath)
    if (!directory.exists()) {
        directory.mkdirs()
    }
    val filePath = "$dirPath/$frameNumber" + "litInfo"

    var out: ObjectOutput? = null
    try {
        out = ObjectOutputStream(FileOutputStream(File(filePath)))
        out.writeObject(lit_info)
        out.close()
    } catch (e: java.lang.Exception) {
        e.printStackTrace()
    }
}


internal fun readVideoFrame(videoNumber: String, frameNumber: String): Bitmap {
    val f = File(FileSource.ORIGINAL_FRAME_PATH + "/" + videoNumber, "$frameNumber.png")
    return BitmapFactory.decodeStream(FileInputStream(f))
}

internal fun readPreviewResultImage(videoNumber: String, frameNumber: String): Bitmap {
    val f = File(FileSource.PREVIEW_RESULT_PATH + "/" + videoNumber, "$frameNumber.jpg")
    return BitmapFactory.decodeStream(FileInputStream(f))
}

internal fun readClothImage(clothNum: Int): Bitmap {
    val f = File(FileSource.CLOTHES_PATH, "$clothNum.jpg")
    return BitmapFactory.decodeStream(FileInputStream(f))
}

internal fun readEdgeImage(clothNum: Int): Bitmap {
    val f = File(FileSource.EDGES_PATH, "$clothNum.jpg")
    return BitmapFactory.decodeStream(FileInputStream(f))
}

internal fun readClothImage(clothUrl: String, contentResolver: ContentResolver): Bitmap {
    /*
    * Converts URL to bitmap */
    val inputStream = contentResolver.openInputStream(Uri.parse(clothUrl))
    val bitmap = BitmapFactory.decodeStream(inputStream)
    Log.i("Image Utils", "Cloth Image is loaded")
    return bitmap
}

internal fun readEdgeImage(clothImage: Bitmap): Bitmap {
    /*
    * Sets transparent region into black
    * and filled region into white */
    val bitmap = Bitmap.createBitmap(
        clothImage.width,
        clothImage.height,
        Bitmap.Config.ARGB_8888
    )
    for (x in 0 until clothImage.width) {
        for (y in 0 until clothImage.height) {
            val pixel = clothImage.getPixel(x, y)
            val alphaValue = android.graphics.Color.alpha(pixel)
            if (alphaValue == 0) {
                bitmap.setPixel(x, y, android.graphics.Color.BLACK)
            } else {
                bitmap.setPixel(x, y, android.graphics.Color.WHITE)
            }
        }
    }
    Log.i("Image Utils", "Edge Image is loaded")
    return bitmap
}

internal fun removeTransparency(clothImage: Bitmap): Bitmap {
    /**
     * Sets transparent region into white
     * and leave filled region as it is */
    val bitmap = Bitmap.createBitmap(
        clothImage.width,
        clothImage.height,
        Bitmap.Config.ARGB_8888
    )
    for (x in 0 until clothImage.width) {
        for (y in 0 until clothImage.height) {
            val pixel = clothImage.getPixel(x, y)
            val alphaValue = android.graphics.Color.alpha(pixel)
            if (alphaValue == 0) {
                bitmap.setPixel(x, y, android.graphics.Color.WHITE)
            } else {
                bitmap.setPixel(x, y, pixel)
            }
        }
    }
    Log.i("ImageUtil", "Cloth Image is loaded")
    return bitmap
}

internal fun readEdgeImage(clothUrl: String, contentResolver: ContentResolver): Bitmap {
    val bitmap = readClothImage(clothUrl, contentResolver)
    return readEdgeImage(bitmap)
}