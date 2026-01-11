# fl-plotter




# SOME RAW NOTES

You've got the right intuition. There's a clear and important distinction between what a styling system (like `.mplstyle` files or Seaborn's `context`) is designed for and what requires a programmatic helper function.

Let's break down your module based on your question.

***

### What Belongs in a Style or `context`?

This system is best for setting **default appearance properties**—the "look" of standard plot elements. From your code, these are prime candidates to be defined in your `gs.mplstyle` file instead of as Python constants:

* **Font Sizes:** This is a classic use case. Your `.mplstyle` file can set these globally.
    * `TITLE_FONT_SIZE = 20` becomes `axes.titlesize: 20`
    * `SUBTITLE_FONT_SIZE = 14` (This is tricky, as there's no "subtitle" concept in Matplotlib, but you can set a default font size).
    * `STANDARD_FONT_SIZE = 14` becomes `font.size: 14` (which would apply to tick labels, legend text, etc.).
* **Legend Styling:** Many of the arguments in your `ax.legend()` call can be set as defaults in the style file.
    * `frameon=False` becomes `legend.frameon: False`
    * `fontsize=14` becomes `legend.fontsize: 14`
    * `columnspacing`, `handlelength`, etc., all have `legend.*` equivalents.
* **Default Figure Size:** You can set a default figure size.
    * `BOXY_FIGURE = (12, 9)` becomes `figure.figsize: 12, 9`


    THE PROBLEM HERE IS THAT I HAVE 2 DEFAULTS, WIDE AND BOXY?? I guess those could be 2 different styles??? do you think that isnthe right way to do it?

The main advantage of putting these in your style sheet is that any standard Matplotlib function (like `ax.set_title()` or `ax.legend()`) will automatically pick them up without you needing to pass arguments.

***

### What's Correctly Handled by Your Helper Functions?

Your intuition is spot-on here. Helper functions are necessary for tasks that involve **logic, content, positioning, or creating non-standard plot elements.**

* **Title & Subtitle Positioning:** **You are absolutely right about this.** A style sheet can set the title's font size and color, but it **cannot** control its precise Y-coordinate (`TITLE_Y_POSITION`). Your `add_chart_titles` function uses `ax.text()` to place text at specific `transAxes` coordinates. This is a structural instruction, not a style, and must be done in a function.
* **Adding a Logo:** This is a perfect helper function. It involves file I/O (`mpimg.imread`), creating a new `axes` object (`fig.add_axes`), and positioning it—all things a style sheet can't do.
* **Complex Legend Logic:** Your `create_legend_at_bottom` function is an excellent example of where a helper is needed. While the *style* of the legend can be set in `.mplstyle`, the *logic*—like converting `Line2D` objects into square patches—is purely programmatic. Furthermore, positioning the legend *below* the plot with `bbox_to_anchor` is a specific layout choice that often needs to be a function call.
* **Adding a Source Citation:** Just like the title, this involves adding new text content at a specific location on the figure (`fig.text`). This is not a styling task.

In short, your functions are correctly handling the **structure**, **content**, and **complex layout**, while the style sheet should handle the default **look and feel**.