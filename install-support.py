# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Installing the slideshow support
# 
# **NOTE** The tools here require IPython 1.0, they do *not* work with
# IPython 0.13.2.
# 
# You should only need to run this notebook *once*. 
# 
# It will install the slideshow tools in your IPython profile from github.  It's done in steps so you can debug it more easily if things go wrong.
# 
# Start by getting the location of your current profile:

# <codecell>

profile_dir = get_ipython().profile_dir.location
profile_dir

# <markdowncell>

# Clone the [Github repo](https://github.com/ipython-contrib/IPython-notebook-extensions) with IPython extensions, including the slideshow support, into the right location:

# <codecell>

import os
tgt = os.path.join( profile_dir, 'static', 'custom')
!git clone https://github.com/ipython-contrib/IPython-notebook-extensions.git $tgt

# <markdowncell>

# Let's `cd` into that directory and check that the contents look right:

# <codecell>

%cd $tgt
!ls

# <markdowncell>

# Finally, write out a `custom.js` file that has activated the slideshow extension.  This is simply the provided `custom.example.js` file, with two lines commented out. Feel free to add more to activate other extensions, as explained in the `README.md` file:

# <codecell>

%%writefile custom.js
// we want strict javascript that fails
// on ambiguous syntax
"using strict";

// do not use notebook loaded  event as it is re-triggerd on
// revert to checkpoint but this allow extesnsion to be loaded
// late enough to work.
//

$([IPython.events]).on('app_initialized.NotebookApp', function(){


    /**  Use path to js file relative to /static/ dir without leading slash, or
     *  js extension.
     *  Link directly to file is js extension.
     *
     *  first argument of require is a **list** that can contains several modules if needed.
     **/

    // require(['custom/noscroll']);
    // require(['custom/clean_start'])
    // require(['custom/toggle_all_line_number'])
    // require(['custom/gist_it']);

    /**
     *  Link to entrypoint if extesnsion is a folder.
     *  to be consistent with commonjs module, the entrypoint is main.js
     *  here youcan also trigger a custom function on load that will do extra
     *  action with the module if needed
     **/
     require(['custom/slidemode/main'],function(slidemode){
    //     // do stuff
     })

});

# <markdowncell>

# That's it! You should now restart your notebook server and reload the pages just to make sure you get fresh CSS.  If everything went well, your toolbar should look like this, with a new button highlighted here in red:
# 
# ![img](files/toolbar-slideshow.png)
# 
# That new button is the toggle to enter live slideshow mode, which you can use to switch between the normal editing mode (with the whole notebook as one long scrolling document) and the presentation mode.
# 
# Now that you've read this, look at the accompanying `notebook-slideshow-example` notebook as a starting illustration of how you can write a presentation-oriented notebook.

