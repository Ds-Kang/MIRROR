package ai.onnxruntime.example.virtualtryon.core

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.net.Uri
import android.provider.MediaStore
import android.util.Log

object ImageHandler {
    private val queue = mutableListOf<Uri>()
    var running = false
    var currentImage: Uri? = null

    fun add(imagePath: Uri) {
        queue.add(imagePath)
    }

    fun remove(index: Int) {
        queue.removeAt(index)
    }

    fun pop(): Uri? {
        return if (queue.isNotEmpty()) {
            queue[0]
            queue.removeAt(0)
        } else {
            null
        }
    }

    fun clear() {
        queue.clear()
    }

    fun isEmpty(): Boolean {
        return queue.isEmpty()
    }

    fun cropImage(context: Context, imageName: String) {
        /**
         * Crops cloth image based on the edge image
         * and saves it to external MediaStore
         * This method is for development */

        // Load the images
        val clothPath = "${FileSource.CLOTHES_PATH}/$imageName.jpg"
        val edgePath = "${FileSource.EDGES_PATH}/$imageName.jpg"
        Log.d("ImageHandler", "clothPath: $clothPath")
        Log.d("ImageHandler", "edgePath: $edgePath")
        val clothBitmap = BitmapFactory.decodeFile(clothPath)
        val edgeBitmap = BitmapFactory.decodeFile(edgePath)

        // Create a mutable bitmap for the cropped image
        val croppedBitmap = Bitmap.createBitmap(clothBitmap.width, clothBitmap.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(croppedBitmap)

        // Iterate over each pixel
        for (x in 0 until clothBitmap.width) {
            for (y in 0 until clothBitmap.height) {
                val edgePixel = edgeBitmap.getPixel(x, y)
                val redValue = Color.red(edgePixel)
                val greenValue = Color.green(edgePixel)
                val blueValue = Color.blue(edgePixel)

                // Calculate the average to determine the brightness
                val averageBrightness = (redValue + greenValue + blueValue) / 3

                // Include the pixel if its brightness is closer to white than black
                if (averageBrightness > 128) {
                    val clothPixel = clothBitmap.getPixel(x, y)
                    croppedBitmap.setPixel(x, y, clothPixel)
                }
            }
        }

        // Save the cropped image to MediaStore
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "cropped_image.png")
            put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
            put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/")
        }

        try {
            context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)?.also { uri ->
                context.contentResolver.openOutputStream(uri).use { outputStream ->
                    croppedBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
                    outputStream?.flush()
                    outputStream?.close()
                }
            }
            Log.d("ImageHandler", "Cropped image saved to MediaStore")
        } catch (e: Exception) {
            Log.e("ImageHandler", "Failed to save cropped image to MediaStore")
            e.printStackTrace()
        }
    }
}