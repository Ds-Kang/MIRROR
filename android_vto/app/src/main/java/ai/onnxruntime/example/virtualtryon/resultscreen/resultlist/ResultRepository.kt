package ai.onnxruntime.example.virtualtryon.resultscreen.resultlist

import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Build
import android.util.Log
import android.util.Size
import androidx.annotation.RequiresApi
import java.io.File

data class Result (
    val file: File,
    val thumbnail: Bitmap,
)
class ResultRepository {

    private val resultList = mutableListOf<Result>()
    private val thumbnailSize = Size(540, 960)

    @RequiresApi(Build.VERSION_CODES.Q)
    fun getResults(source: File): List<Result> {
        resultList.clear()
        val files = source.list()

        for (fileDir in files!!) {
            val file = File(source, fileDir)
            // All other videos are in the root directory
            if (file.isDirectory) {
                for (videoFile in file.listFiles()!!) {
                    try {
                        val thumbnail = ThumbnailUtils.createVideoThumbnail(
                            File(videoFile.absolutePath),
                            thumbnailSize,
                            null
                        )
                        resultList.add(Result(videoFile, thumbnail))
                    }
                    catch (e: Exception) {
                        Log.e("VideoRepository", "File open failed: $file/$videoFile")
                    }
                }
            } else {
                Log.d("VideoRepository", "Skipping file: $file")
            }
        }
        return resultList
    }
}