package ai.onnxruntime.example.virtualtryon.resultscreen.resultlist

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import java.io.File

class ResultListViewModel : ViewModel() {
    private val resultRepository = ResultRepository()

    private val _results = MutableLiveData<List<Result>>().apply {
        value = emptyList()
    }
    val results: MutableLiveData<List<Result>> = _results

    @RequiresApi(Build.VERSION_CODES.Q)
    fun getResults(source: File) {
        _results.value = resultRepository.getResults(source)
    }

    fun deleteResult(position: Int) {
        val results = _results.value!! as MutableList<Result>
        results[position].file.delete()
        results.removeAt(position)
    }
}