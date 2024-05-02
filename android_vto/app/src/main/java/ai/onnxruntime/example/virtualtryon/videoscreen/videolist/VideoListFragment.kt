package ai.onnxruntime.example.virtualtryon.videoscreen.videolist

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.annotation.RequiresApi
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.RecyclerView
import ai.onnxruntime.example.virtualtryon.R
import ai.onnxruntime.example.virtualtryon.core.FileSource
import java.io.File

class VideoListFragment : Fragment() {

    private lateinit var safeContext: Context
    private lateinit var videoRepository: VideoRepository

    override fun onAttach(context: Context) {
        super.onAttach(context)
        safeContext = context
        videoRepository = VideoRepository()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_videos, container, false)
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val viewModel = ViewModelProvider(requireActivity())[VideoListViewModel::class.java]
        viewModel.getVideo() // Initialize the video list

        viewModel.videos.observe(viewLifecycleOwner) {
            // Bind the video list to the view
            val viewAdapter = VideosListAdapter(it as MutableList<Video>, viewModel)
            view.findViewById<RecyclerView>(R.id.videos_list).run {
                setHasFixedSize(true)
                adapter = viewAdapter
            }
            // When there is no video, show the no video text
            if (it.isEmpty()) {
                view.findViewById<View>(R.id.no_video).visibility = View.VISIBLE
            } else {
                view.findViewById<View>(R.id.no_video).visibility = View.GONE
            }
        }
    }
}
