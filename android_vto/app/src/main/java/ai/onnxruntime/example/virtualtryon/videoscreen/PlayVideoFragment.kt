package ai.onnxruntime.example.virtualtryon.videoscreen

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.VideoView
import androidx.fragment.app.Fragment
import ai.onnxruntime.example.virtualtryon.R

class PlayVideoFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_play_video, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val videoView = view.findViewById<VideoView>(R.id.video_view)
        val videoPath = arguments?.getString("videoPath")

        try {
            videoView.setVideoPath(videoPath)
            videoView.start()
        } catch (e: Exception) {
            e.printStackTrace()
        }

    }
}
