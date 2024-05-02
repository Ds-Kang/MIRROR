package ai.onnxruntime.example.virtualtryon.videoscreen.videolist

import ai.onnxruntime.example.virtualtryon.core.FileSource
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import java.io.File

class VideoListViewModel : ViewModel() {
    private val videoRepository = VideoRepository()

    // Video list
    private val _videos = MutableLiveData<List<Video>>().apply {
        value = emptyList()
    }
    val videos: MutableLiveData<List<Video>> = _videos

    @RequiresApi(Build.VERSION_CODES.Q)
    fun getVideo() {
        _videos.value = videoRepository.getVideos()
    }

    private fun setVideo(videoList: List<Video>) {
        _videos.value = videoList
    }

    fun setVideoAsDefault(position: Int) {
        val videos = _videos.value!! as MutableList<Video>
        val previousDefaultVideo = videos.find { it.isDefault }!!
        previousDefaultVideo.isDefault = false

        videos[position].isDefault = true
        videoRepository.setDefaultVideo(videos[position].file)
        videos.sortBy { !it.isDefault }

        setVideo(videos)
    }

    fun deleteVideo(position: Int) {
        // Delete the frames and preview results first
        val videos = _videos.value!! as MutableList<Video>
        val videoName = videos[position].file.name.substringBefore(".mp4")
        val frames = File(FileSource.ORIGINAL_FRAME_PATH, videoName)
        val previewResults = File(FileSource.PREVIEW_RESULT_PATH, videoName)
        val dnnInfo = File(FileSource.DNN_INFO_PATH, videoName)
        val litInfo = File(FileSource.LIT_INFO_PATH, videoName)
        frames.deleteRecursively()
        previewResults.deleteRecursively()
        dnnInfo.deleteRecursively()
        litInfo.deleteRecursively()

        // Delete the video file

        videos[position].file.delete()
        if (videos[position].isDefault) {
            videoRepository.setDefaultVideo()
        }
        videos.removeAt(position)
        _videos.value = videos
    }
}