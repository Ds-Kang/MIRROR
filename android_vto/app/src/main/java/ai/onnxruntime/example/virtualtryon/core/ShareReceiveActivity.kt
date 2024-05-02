package ai.onnxruntime.example.virtualtryon.core

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager

class ShareReceiveActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (intent.action != Intent.ACTION_SEND || intent.type == null) {
            Log.e("ShareReceiveActivity", "Invalid intent")
            finish()
            return
        }

        val imageUri = intent.clipData?.getItemAt(0)?.uri ?: intent.getParcelableExtra(Intent.EXTRA_STREAM)
        Log.d("ShareReceiveActivity", "Received image from share intent: $imageUri")

        if (imageUri == null) {
            Toast.makeText(this, "No image received", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        runRuntimeService(imageUri.toString())

        Toast.makeText(this, "Image sent to MIRROR", Toast.LENGTH_SHORT).show()
        finish()
    }

    private fun runRuntimeWorker(clothImageUrl: String) {
        val data = Data.Builder()
            .putString("type", "runtime")
            .putString("clothImageUrl", clothImageUrl)
            .build()

        val runtimeRequest = OneTimeWorkRequestBuilder<InferenceWorker>()
            .setInputData(data)
            .build()

        WorkManager.getInstance(this).beginUniqueWork(
            "InferenceWorker",
            ExistingWorkPolicy.REPLACE,
            runtimeRequest
        ).enqueue()
    }

    private fun runRuntimeService(clothImageUrl: String) {
        val intent = Intent(this, InferenceService::class.java)
        intent.putExtra("type", "runtime")
        intent.putExtra("clothImageUrl", clothImageUrl)
        startForegroundService(intent)
    }
}