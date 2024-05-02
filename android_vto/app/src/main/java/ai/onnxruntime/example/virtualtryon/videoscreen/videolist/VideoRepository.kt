package ai.onnxruntime.example.virtualtryon.videoscreen.videolist

import android.graphics.Bitmap
import android.media.ThumbnailUtils
import android.os.Build
import android.util.Log
import android.util.Size
import androidx.annotation.RequiresApi
import ai.onnxruntime.example.virtualtryon.core.FileSource
import java.io.File

data class Video(
    val file: File,
    val thumbnail: Bitmap,
    var isDefault: Boolean = false
)

class VideoRepository {

    private val videoList = mutableListOf<Video>()
    private val thumbnailSize = Size(540, 960)

    init {
        val defaultVideoDir = File(FileSource.ORIGINAL_VIDEO_PATH, "default")
        if (!defaultVideoDir.exists()) {
            defaultVideoDir.mkdir()
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    fun getVideos(): List<Video> {
        val source = File(FileSource.ORIGINAL_VIDEO_PATH)
        videoList.clear()
        val files = source.list()
        for (file in files!!) {
            Log.d("VideoRepository", "File found: $file")
        }

        for (fileDir in files) {
            val file = File(source, fileDir)
            // Default video is the first video in the default directory
            if (file.isDirectory && fileDir == "default") {
                try {
                    val defaultVideo = File(file, file.list()!![0])
                    val thumbnail = ThumbnailUtils.createVideoThumbnail(defaultVideo, thumbnailSize, null)
                    videoList.add(Video(defaultVideo, thumbnail, true))
                } catch (e: IndexOutOfBoundsException) {
                    Log.e("VideoRepository", "No default video found")
                    if (setDefaultVideo()) {
                        Log.d("VideoRepository", "Default video set")
                        getVideos()
                    } else {
                        Log.d("VideoRepository", "No video found")
                        break
                    }
                }
            }
            // All other videos are in the root directory
            else if (file.isFile) {
                Log.d("VideoRepository", "File found: $file")
                val thumbnail = ThumbnailUtils.createVideoThumbnail(file, thumbnailSize, null)
                videoList.add(Video(file, thumbnail, false))
            }
        }
        videoList.sortBy { !it.isDefault }  // Default video goes first
        return videoList
    }

    /**
     * When there is no default video,
     * change the first video into default */
    fun setDefaultVideo(): Boolean {
        val source = File(FileSource.ORIGINAL_VIDEO_PATH)
        val files = source.listFiles()
        val defaultDir = File(source, "default")
        for (file in files!!) {
            if (file.isFile) {
                file.renameTo(File(defaultDir, file.name))
                return true
            }
        }
        return false
    }

    /**
     * Make the selected video as default */
    fun setDefaultVideo(file: File) {
        val source = File(FileSource.ORIGINAL_VIDEO_PATH)
        val defaultDir = File(source, "default")
        if (defaultDir.exists()) {
            val defaultVideo = File(defaultDir, defaultDir.list()!![0])
            if (defaultVideo.exists()) {
                defaultVideo.renameTo(File(source, defaultVideo.name))
            }
            file.renameTo(File(defaultDir, file.name))
        }
    }

    /**
     * For debugging
     * Deletes all videos including the default */
    private fun deleteAllVideos() {
        val source = File(FileSource.ORIGINAL_VIDEO_PATH)
        val files = source.listFiles()
        for (file in files!!) {
            if (file.isFile) {
                file.delete()
            }
            if (file.isDirectory && file.name == "default") {
                for (video in file.listFiles()!!) {
                    video.delete()
                }
            }
        }
    }
}