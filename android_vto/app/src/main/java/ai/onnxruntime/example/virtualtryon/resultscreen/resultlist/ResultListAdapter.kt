package ai.onnxruntime.example.virtualtryon.resultscreen.resultlist

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.PopupMenu
import androidx.navigation.findNavController
import androidx.recyclerview.widget.RecyclerView
import ai.onnxruntime.example.virtualtryon.R
import android.util.Log
import android.widget.TextView
import java.io.File

class ResultsListAdapter(
    private val results: MutableList<Result>,
    private val viewModel: ResultListViewModel,
    private val resultSource: File
) : RecyclerView.Adapter<ResultsListAdapter.ViewHolder>() {
    class ViewHolder(val item: View) : RecyclerView.ViewHolder(item)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val itemView = LayoutInflater.from(parent.context).inflate(R.layout.video_thumbnail, parent, false)
        return ViewHolder(itemView)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {

        // Bind thumbnail to the view
        holder.item.findViewById<ImageView>(R.id.video_thumbnail_image).setImageBitmap(
            results[position].thumbnail
        )

        val defaultMark = holder.item.findViewById<TextView>(R.id.video_thumbnail_default)
        defaultMark.visibility = View.GONE

        // Set thumbnail size
        val thumbnailMarginDP = R.dimen.video_thumbnail_margin
        val screenWidthDP = holder.item.context.resources.displayMetrics.widthPixels
        val thumbnailWidth = (screenWidthDP - 3 * holder.item.context.resources.getDimension(thumbnailMarginDP)) / 2
        val thumbnailHeight = thumbnailWidth * 16 / 9
        holder.item.layoutParams.width = thumbnailWidth.toInt()
        holder.item.layoutParams.height = thumbnailHeight.toInt()

        // Add bottom margin to the last items
        val layoutParams = holder.item.layoutParams as ViewGroup.MarginLayoutParams
        if (position >= results.size - 2 + results.size % 2) {
            layoutParams.bottomMargin = holder.item.context.resources.getDimension(thumbnailMarginDP).toInt()
        } else {
            layoutParams.bottomMargin = 0
        }
        holder.item.layoutParams = layoutParams

        // Set the onClickListener to each item
        holder.item.setOnClickListener {
            val bundle = Bundle()
            bundle.putString("resultPath", results[position].file.absolutePath)
            holder.item.findNavController().navigate(
                R.id.action_results_to_play,
                bundle
            )
        }

        // Set the onClickListener to the menu button
        holder.item.findViewById<ImageView>(R.id.video_thumbnail_menu).setOnClickListener {
            val popupMenu = PopupMenu(holder.item.context, it)
            popupMenuClickListener(popupMenu, position)
        }
    }

    private fun popupMenuClickListener(popupMenu: PopupMenu, position: Int) {
        popupMenu.menuInflater.inflate(R.menu.popup_fitting, popupMenu.menu)
        popupMenu.show()

        popupMenu.apply {
            setOnMenuItemClickListener { menuItem ->
                when (menuItem.itemId) {
                    R.id.menu_delete -> {
                        viewModel.deleteResult(position)
                        notifyItemRemoved(position)
                        true
                    }
                    else -> false
                }
            }
        }
    }

    // Return the size of your dataset (invoked by the layout manager)
    override fun getItemCount() = results.size
}