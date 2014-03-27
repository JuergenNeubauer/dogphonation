# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <p class="title">Title here: a Notebook-based talk</p>
# 
# <p class="subtitle">With a great subtitle!</p>
# 
# <center>
# 
# <p class="gap05"<p>
# <h2>[ipython.org](http://ipython.org)</h2>
# 
# <p class="gap05"<p>
# <h3> Fernando Pérez </h3>
# <h3>[fperez.org](http://fperez.org), [@fperez_org](http://twitter.com/fperez_org)</h3>
# <h3>U.C. Berkeley</h3>
# 
# <p class="gap2"<p>
# </center>

# <markdowncell>

# ## How to use this to write slideshows...
# 
# ### Installation check
# 
# First, this presumes you've already read and executed once the accompanying `install-support` notebook, which should install the necessary tools for you, and you restarted your notebook server. If everything went well, right now your toolbar should look like this, with a new button highlighted here in red:
# 
# ![img](files/toolbar-slideshow.png)
# 
# That new button is the toggle to enter live slideshow mode, which you can use to switch between the normal editing mode (with the whole notebook as one long scrolling document) and the presentation mode.
# 
# ### Loading the CSS
# 
# All the CSS is kept in a file called `style.css`, and it's loaded, along with a few handy utilities, by the `talktools.py` script.  Simply run it once to load everything, or if you make any tweaks to the CSS:

# <codecell>

# I keep this as a cell in my title slide so I can rerun 
# it easily if I make changes, but it's low enough it won't
# be visible in presentation mode.
%run talktools

# <codecell>

website('nbviewer.ipython.org', 'Notebook Viewer - nbviewer')

# <headingcell level=2>

# Editing

# <markdowncell>

# Once you've loaded the above, the *editing* workflow (a bit primitive, admittedly) is:
# 
# * From the "Cell Toolbar" menu, select "Slideshow". This will give you a little dropdown for each cell.
# 
# * To start a new slide, mark a cell as "Slide".
# 
# * If you want a chunk of a slide to be revealed incrementally, mark it as 'Fragment'.
# 
# * See the source for the various slides in this file or the contents of the `style.css` file for various useful CSS classes that you can use.
# 
# 
# 
# **Note** One thing this mode does NOT have is any notion of page size. What I do to estimate the size of my slides is to resize my browser window so the slideshow area is ~ 760px tall.  Since most projectors work at 1024x768, this works well.  I try to keep most of them confined to that vertical size, though if needed it's still OK to have taller ones, you just need to remember to scroll them down during presentation.

# <headingcell level=2>

# Presenting

# <markdowncell>

# For presentations, you should toggle the "Cell toolbar" menu to "None", so that it doesn't appear in your slides.  Then you can move back and forth with the GUI controls.
# 
# If you click on the "Enable Slide Mode" button, the right/left arrows will become also keys to move between slides.  But note that at any point if you start typing into the notebook, that functionality will be inactivated, so you can use the arrows normally.  You can turn it on again for further presentation by clicking the button again.

# <headingcell level=2>

# Example slides

# <markdowncell>

# The next slides are a few example ones from a real talk of mine, so you can get a sense for how to lay things out in a presentation.

# <markdowncell>

# <div class="slide-header">Why IPython?</div>
# 
# <center>
# *"The purpose of computing is insight, not numbers"*
# <p style="margin-left:70%">Hamming '62</p>
# </center>

# <markdowncell>

# ## The Lifecycle of a Scientific Idea (schematically)
# 
# 1. <span class="emph">**Individual**</span> exploratory work
# 2. <span class="emph">**Collaborative**</span> development
# 3. <span class="emph">**Parallel**</span> production runs (HPC, cloud, ...)
# 4. <span class="emph">**Publication**</span> (with <span class="warn">reproducible</span> results!)
# 5. <span class="emph">**Education**</span>
# 6. Goto 1.

# <markdowncell>

# <div class="slide-header">From a better shell...</div>
# <center>
# <img src="files/ipython_console4.png" width="80%">
# </center>

# <markdowncell>

# <div class="slide-header">... to sharing notebooks with zero-install...</div>
# ## Matthias Bussonnier, 2012

# <codecell>

website('nbviewer.ipython.org', 'Notebook Viewer - nbviewer')

# <markdowncell>

# <div class="slide-header">... to the first White House Hackathon...</div>
# ## IPython and NetworkX go to DC

# <codecell>

YouTubeVideo('sjfsUzECqK0', start='40', width=600, height=400)

# <markdowncell>

# 
# # `IPython.parallel`
# 
# * Cluster: one *controller* (master), N dynamic *engines* (slaves). $N \lesssim 300$.
# * Accessed by users via dynamic proxy: `parallel.Client`.
# * A `Client` creates as many *views* as desired by user:
#   - `DirectView`: SPMD.
#   - `LoadBalancedView`: automatic task distribution.
#   - Blocking and async APIs.
#   - Direct `put` and `get` (and `scatter`/`gather`).
#   - Views can span arbitrary subgroups of the cluster.
#   - Multiple views coexist simultaneously.
# * One-sided communication model.
# * Happy coexistence with MPI.
# * All comms with [ØMQ](http://www.zeromq.org), in C++, no GIL, zero-copy when possible.

# <markdowncell>

# # Lessons in building a community
# 
# ## A delicate balancing act
# 
# * Individual leadership and sense of ownership vs turf wars.
# * Clear project vision vs. broad engagement of a community with many ideas.
# 
# ## Disagreement: highly reactive fuel
# 
# * Can power creativity or spectacularly blow up.
# * Patience, trust, generosity and respect.
# * Make calls and live with the mistakes without recrimination.
# 
# ## The rules apply to everyone, starting with me
# 
# * Even if they slow me down!
# 

