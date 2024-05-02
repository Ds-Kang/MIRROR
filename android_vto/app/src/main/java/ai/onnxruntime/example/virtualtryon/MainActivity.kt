package ai.onnxruntime.example.virtualtryon

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.NavController
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import ai.onnxruntime.example.virtualtryon.core.FileSource
import ai.onnxruntime.example.virtualtryon.core.VideoHandler
import android.Manifest
import android.app.NotificationChannel
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions
import androidx.annotation.RequiresApi
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.bottomnavigation.BottomNavigationView
import org.opencv.video.Video

class MainActivity : AppCompatActivity() {

    private lateinit var navController: NavController
    private lateinit var appBarConfiguration: AppBarConfiguration

    init {
        System.loadLibrary("native_lib")
    }

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        checkPermission()

        // Hide action bar
        if (supportActionBar != null) {
            supportActionBar!!.hide()
        }

        // Set up notification channel
        val notificationChannel = NotificationChannel("ort_virtual_try_on", "MIRROR", android.app.NotificationManager.IMPORTANCE_DEFAULT)
        val notificationManager = getSystemService(android.app.NotificationManager::class.java)
        notificationManager.createNotificationChannel(notificationChannel)

        // Set up IO directories
        FileSource.initializeRootDir(applicationContext.filesDir.absolutePath)
        Log.i("MainActivity", "Original file dir: ${FileSource.ORIGINAL_PATH}")

        // Fragment container
        val navHostFragment = supportFragmentManager.findFragmentById(
            R.id.nav_host_container
        ) as NavHostFragment
        navController = navHostFragment.navController

        // Setup the bottom navigation view with navController
        val bottomNavigationView = findViewById<BottomNavigationView>(R.id.bottom_nav)
        bottomNavigationView.setupWithNavController(navController)

        // Setup the ActionBar with navController and 3 top level destinations
        appBarConfiguration = AppBarConfiguration(
            setOf(R.id.results_nav, R.id.camera_nav, R.id.videos_nav)
        )
        setupActionBarWithNavController(navController, appBarConfiguration)
    }

    override fun onSupportNavigateUp(): Boolean {
        return navController.navigateUp(appBarConfiguration)
    }

    //권한 확인

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    private fun checkPermission() {
        val requestPermissions = registerForActivityResult(RequestMultiplePermissions()) { results ->
            if (results[Manifest.permission.READ_EXTERNAL_STORAGE] == true &&
                results[Manifest.permission.WRITE_EXTERNAL_STORAGE] == true &&
                results[Manifest.permission.READ_MEDIA_IMAGES] == true &&
                results[Manifest.permission.POST_NOTIFICATIONS] == true) {
                Log.i(TAG, "All permissions granted")
            } else {
                Log.e(TAG, "Some permissions are not granted")
            }
        }
        requestPermissions.launch(
            arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.POST_NOTIFICATIONS
            )
        )
    }

    private fun checkPermission(permission: String) {
        val permissionCheck = ContextCompat.checkSelfPermission(this, permission)
        if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(permission), PERMISSION_REQUEST_CODE)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        const val TAG = "ORTVirtualTryOn"
        const val PERMISSION_REQUEST_CODE = 1001
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
    }
}
