package ai.onnxruntime.example.virtualtryon.resultscreen.resultlist

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.util.Log
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

class ResultListFragment : Fragment() {

    private lateinit var safeContext: Context
    private lateinit var resultRepository: ResultRepository
    private lateinit var viewModel: ResultListViewModel

    override fun onAttach(context: Context) {
        super.onAttach(context)
        safeContext = context
        resultRepository = ResultRepository()
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment
        return inflater.inflate(R.layout.fragment_results, container, false)
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        viewModel = ViewModelProvider(requireActivity())[ResultListViewModel::class.java]
        val resultSource = getProcessedVideoSource()
        viewModel.getResults(resultSource)

        viewModel.results.observe(viewLifecycleOwner) {
            // Bind the result list to the view
            val viewAdapter = ResultsListAdapter(it as MutableList<Result>, viewModel, resultSource)
            view.findViewById<RecyclerView>(R.id.results_list).run {
                setHasFixedSize(true)
                adapter = viewAdapter
            }
            // When there is no video, show the no video text
            if (it.isEmpty()) {
                view.findViewById<View>(R.id.no_result).visibility = View.VISIBLE
            } else {
                view.findViewById<View>(R.id.no_result).visibility = View.GONE
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    override fun onResume() {
        super.onResume()
        viewModel.getResults(getProcessedVideoSource())
    }

    private fun getProcessedVideoSource(): File {
        return File(FileSource.RUNTIME_RESULTS_PATH)
    }
}
