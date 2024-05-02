package ai.onnxruntime.example.virtualtryon.resultscreen

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.VideoView
import androidx.fragment.app.Fragment
import ai.onnxruntime.example.virtualtryon.R

class PlayResultFragment : Fragment() {
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_play_result, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val resultView = view.findViewById<VideoView>(R.id.result_view)
        val resultPath = arguments?.getString("resultPath")

        try {
            resultView.setVideoPath(resultPath)
            resultView.start()
        } catch (e: Exception) {
            e.printStackTrace()
        }

    }
}
