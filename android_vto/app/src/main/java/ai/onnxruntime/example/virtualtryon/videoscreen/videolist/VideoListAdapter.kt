package ai.onnxruntime.example.virtualtryon.videoscreen.videolist

import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.PopupMenu
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.navigation.findNavController
import androidx.recyclerview.widget.RecyclerView
import ai.onnxruntime.example.virtualtryon.R
import java.io.File

@RequiresApi(Build.VERSION_CODES.O)
class VideosListAdapter(
    private val videos: MutableList<Video>,
    private val viewModel: VideoListViewModel
) : RecyclerView.Adapter<VideosListAdapter.ViewHolder>() {
    class ViewHolder(val item: View) : RecyclerView.ViewHolder(item)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val itemView = LayoutInflater.from(parent.context).inflate(R.layout.video_thumbnail, parent, false)
        return ViewHolder(itemView)
    }

    // Fills in the view with the data
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {

        // Bind the video thumbnail to the view
        holder.item.findViewById<ImageView>(R.id.video_thumbnail_image).setImageBitmap(
            videos[position].thumbnail
        )

        // Set thumbnail size
        val thumbnailMarginDP = R.dimen.video_thumbnail_margin
        val screenWidthDP = holder.item.context.resources.displayMetrics.widthPixels
        val thumbnailWidth = (screenWidthDP - 3 * holder.item.context.resources.getDimension(thumbnailMarginDP)) / 2
        val thumbnailHeight = thumbnailWidth * 16 / 9
        holder.item.layoutParams.width = thumbnailWidth.toInt()
        holder.item.layoutParams.height = thumbnailHeight.toInt()

        // Add bottom margin to the last items
        val layoutParams = holder.item.layoutParams as ViewGroup.MarginLayoutParams
        if (position >= videos.size - 2 + videos.size % 2) {
            layoutParams.bottomMargin = holder.item.context.resources.getDimension(thumbnailMarginDP).toInt()
        } else {
            layoutParams.bottomMargin = 0
        }
        holder.item.layoutParams = layoutParams

        // Show the default video mark if the video is the default video
        if (videos[position].isDefault) {
            holder.item.findViewById<TextView>(R.id.video_thumbnail_default).visibility = View.VISIBLE
        } else {
            holder.item.findViewById<TextView>(R.id.video_thumbnail_default).visibility = View.GONE
        }

        // Set the onClickListener to each video thumbnail
        holder.item.setOnClickListener {
            val bundle = Bundle()
            bundle.putString("videoPath", videos[position].file.absolutePath)
            holder.item.findNavController().navigate(
                R.id.action_videos_to_play,
                bundle
            )
        }

        // Handle the popup menu button
        holder.item.findViewById<ImageView>(R.id.video_thumbnail_menu).setOnClickListener {
            val popupMenu = PopupMenu(holder.item.context, it)
            popupMenuClickListener(popupMenu, position)
        }
    }

    private fun popupMenuClickListener(popupMenu: PopupMenu, position: Int) {
        popupMenu.menuInflater.inflate(R.menu.popup_videos, popupMenu.menu)
        popupMenu.show()

        popupMenu.apply {
            setOnMenuItemClickListener { menuItem ->
                when (menuItem.itemId) {
                    R.id.menu_delete -> {
                        viewModel.deleteVideo(position)
                        notifyItemRemoved(position)
                        true
                    }
                    R.id.menu_set_default -> {
                        viewModel.setVideoAsDefault(position)
                        true
                    }
                    else -> false
                }
            }
        }
    }

    override fun getItemCount() = videos.size
}