# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <h1 class="title">Why you should write buggy software with as few features as possible</h1>

# <markdowncell>

# <center>
# 
# <h2>Brian Granger (ellisonbg)</h2>
# 
# <div>
#     <img class="logo" src="files/images/calpoly_logo.png" height=100 />
# </div>
# 
# <div>
#     <img class="logo" src="files/images/logo.png" height=100 />
# </div>
# 
# <h3>SciPy, June 2013</h3>
# </center>

# <headingcell level=1>

# Background

# <markdowncell>

# * In the summer of 2011, I locked myself in our basement and wrote the first version of the 6th incarnation the IPython Notebook
# * It was full of horrible, annoying bugs
# * It lacked many features that we consider to be absolutely necessary
# * This was a deliberate choice we made based on our experience with the first 5 incarnations
# * Almost overnight, it was broadly adopted by the community and has become a popular and productive tool
# * It still has many of these bugs and lacks many of the needed features
# * This has challenged our thinking about software development

# <markdowncell>

# <h1 class="bigtitle">Theory</h1>

# <headingcell level=1>

# A theory of software engineering

# <markdowncell>

# Here is one possible theory of software engineering that we are tempted by:
# 
# 1. Features are good
# 2. Bugs are bad
# 3. Therefore, the ideal software will have lots of features and few bugs
# 4. The number of features and bugs is only limited by developer time
# 5. You shouldn't release software until it has lots of features and few bugs

# <headingcell level=1>

# Does this theory generalize?

# <markdowncell>

# Let's try to generalize this theory to food:
# 
# 1. Ice cream is good
# 2. Vegatables are bad
# 3. Therefore, the ideal diet will involve lots of ice cream and few vegetables
# 4. The amount of ice cream involved is only limited by how much money you have

# <headingcell level=1>

# My 7 year old son likes this theory

# <markdowncell>

# <center>
# <img src="files/images/reed.jpeg" width=700 />
# </center>

# <headingcell level=1>

# Problems with this theory

# <markdowncell>

# * Developer time is a finite resource
# * Active projects are flooded (thanks GitHub!) with Pull Requests and Issues:
#   - NumPy: 763 Issues, 42 Pull Requests
#   - SciPy: 581 Issues, 28 Pull Requests
#   - IPython: 592 Issues, 16 Pull Requests
#   - Matplotlib: 323 Issues, 44 Pull Requests
#   - SymPy: 1142 Issues, 121 Pull Requests
# * Reality is more complicated:
#   - Features have hidden costs
#   - Bugs have hidden benefits
#   - We love software that has few features (for example Twitter)
# 
# ### We need a rational process for deciding how to spend our time as developers. Which new features do we add? Which bugs do we fix? How do we prioritize these things?

# <markdowncell>

# <h1 class="bigtitle">The hidden costs of features</h1>

# <markdowncell>

# <h3 class="point">Each new feature adds complexity to the code base</h3>
# <h3 class="point">Complexity makes a code base less hackable, maintainable, extensible</h3>

# <markdowncell>

# <h3 class="point">Each new feature increases the "bug surface" of the project</h3>
# <h3 class="point">When a feature also adds complexity, those bugs become harder to find and fix</h3>

# <markdowncell>

# <h3 class="point">Each new feature requires documentation to be written and maintained</h3>

# <markdowncell>

# <h3 class="point">Each new feature requires support over email/IRC/HipChat</h3>

# <markdowncell>

# <h3 class="point">Endless feature expansion, or feature creep, requires developers to specialize</h3>
# 
# <h3 class="point">Individuals can't follow the entire project, so they have to focus on a subset that can fit into their brain and schedule</h3>

# <markdowncell>

# <h3 class="point">Each new feature has to be tested on a wide variety on platforms (Linux, Mac, Windows) and environments (PyPy, Python 2, Python 3)</h3>

# <markdowncell>

# <h3 class="point">Each new feature adds complexity to the user experience</h3>
# <h3 class="point">Sometimes it's the documentation or API, other times the UI or configuration options</h3>
# <h3 class="point">This increases the cognitive load on your users</h3>

# <markdowncell>

# <h3 class="point">When you spend on one feature, another feature or bug fix didn't get worked on</h3>
# <h3 class="point">If you didn't prioritize things beforehand, you just spent time on something less important to your users</h3>

# <markdowncell>

# <h3 class="point">Features multiply like bunnies</h3>
# 
# <h3 class="point">"wow, that new feature Y is really cool, could you make it do X as well?"</h3>

# <markdowncell>

# <h3 class="point">Features are easy to add, difficult to remove</h3>
# <h3 class="point">Once you add a feature, you are stuck with the costs and liabilities</h3>

# <markdowncell>

# <h3 class="point">I am not suggesting that features are bad, only that they have costs that need to be counted</h3>

# <headingcell level=1>

# Features: IPython Notebook

# <markdowncell>

# We have learned a lot about features in developing the IPython Notebook.
# 
# ## Features we said no to initially:
# 
# * Multi-directory navigation
# * Multi-user capabilities
# * Security
# * URLs with Notebook paths and names (`path/to/my/notebook.ipynb`)
# * Autosave/checkpointing
# 
# It was a deliberate choice for us to leave these features out of the Notebook initially. This was extremely hard to for us to do - emotionally and psychologically. But it was one of the best things we did as it enabled us to move quickly on more important features.
# 
# ## Features we have always said no to:
# 
# * Extensible cell types
# * Lots of cell and notebook metadata
# * Limiting the Notebook's ability to execute arbitrary code for security reasons

# <headingcell level=1>

# Regrets a.k.a. lessons learned

# <markdowncell>

# ## Features we said yes and later no to:
# 
# * XML Notebook format
# * Database backed Notebook server
# * Multiple worksheets within a single Notebook
# * reStructuredText cells
# 
# We spend (literally) months developing these features over two summers. All of that work has been thrown away and we are still suffering from some of these decisions. Some of this could have been prevented had we been more disciplined about the following question:
# 
# ### What is the simplest possible Notebook we can implement that would be useful?
# 
# * Can we implement a Notebook without an XML Notebook format?  Yes!
# * Can we implement a Notebook without a database? Yes!
# * Can we implement a Notebook without worksheets?  Yes!
# * Can we implement a Notebook with the simpler Markdown syntax? Yes!

# <markdowncell>

# <h1 class="bigtitle">The hidden benefits of bugs</h1>

# <markdowncell>

# <h3 class="point">Bugs are a sign that people are using your software</h3>

# <markdowncell>

# <h3 class="point">Bugs tell you how your users are using your software</h3>

# <markdowncell>

# <h3 class="point">Bugs tell you which features are important</h3>

# <markdowncell>

# <h3 class="point">Bugs are opportunities to improve the testing of your software</h3>

# <markdowncell>

# <h3 class="point">Bug reporting/fixing can be a great starting point for new developers</h3>

# <markdowncell>

# <h3 class="point">I am not suggesting that bugs are entirely **good**, only that they serve a useful purpose in actively developed software</h3>

# <headingcell level=1>

# Bugs: IPython Notebook

# <markdowncell>

# Some bugs have taught us useful things:
# 
# * The notebook is broken on IE<10
#   - We quickly learned that almost none of our users were affected
#   - This allowed us to commit to using WebSockets from the start
#   - The result is simple, clean code in the Notebook server and client
# * Jumping scroll bugs
#   - Clicking on the output area of a large cell causes the Notebook area to jumpily scroll the top of the cell into focus
#   - The same thing happens when you run a cell and the next cell focuses
#   - Annoying as hell, but users put up with it...even to this day :(
#   - Fixing these bugs is very subtle and will change the UX in significant ways
#   - Watching people use the Notebook with these bugs has given us invaluable insight about the UX
#   - This knowledge is enabling us to develop a better UX

# <markdowncell>

# <h1 class="bigtitle">Not all features should be implemented</h1>

# <headingcell level=1>

# This requires a cultural solution

# <markdowncell>

# * This necessarily means you are going to have to say "no" to enthusiastic developers and users
# * How can you do this without hurting people's feelings?
# * How do you build this into your community and developer DNA?
# 
# Here are some ideas...

# <markdowncell>

# <h3 class="point">Create a roadmap for the project that describes which features are going to be added and which are not</h3>
# 
# <h3 class="point">Publicize this roadmap, discuss it with developers and make it an important part of the development process</h3>

# <markdowncell>

# <h3 class="point">Decide on a finite scope, or vision, for the project</h3>
# <h3 class="point">Communicate that vision to your community</h3>
# <h3 class="point">Implement features that are within that scope</h3>

# <markdowncell>

# <h3 class="point">Make features fight hard to be accepted and implemented</h3>
# <h3 class="point">Communicate to the community and developers that the default answer to new feature requests is no (it's not personal!)</h3>
# <h3 class="point">Don't even consider implementation until the much of the community is crying "we absolutely must have this." </h3>

# <markdowncell>

# <h3 class="point">Create a workflow that separates new feature requests from other tickets/issues</h3>
# <h3 class="point">When people submit new feature requests, encourage discussion, but don't automatically promote the feature to the project's todo list</h3>

# <markdowncell>

# <h3 class="point">When new feature requests are submitted, discuss the specific costs and liabilities associated with the feature</h3>
# <h3 class="point">Build this thinking into your development DNA</h3>

# <markdowncell>

# <h3 class="point">Communicate to the community why it is important to fight against boundless feature expansion</h3>
# <h3 class="point">Focus on the benefits: smaller, simpler code base, fewer bugs, more time to focus on important features, easier to support, etc.</h3>

# <markdowncell>

# <h3 class="point">Remove features that have too great a cost, are outside your project's scope or that few users actually use</h3>

# <markdowncell>

# <h3 class="point">Refactor the codebase to reduce complexity</h3>
# <h3 class="point">Extra bonus points if you can implement a new feature while reducing the complexity of the code base</h3>

# <markdowncell>

# <h3 class="point">Improve testing</h3>

# <headingcell level=1>

# Summary

# <markdowncell>

# ## Decide on a finite scope for a project and communicate it to the community
# ## Implement a minimal set of features that cover that scope
# ## Ship software with bugs
# ## Use those bugs to learn useful information and attract developers

# <headingcell level=1>

# Resources

# <markdowncell>

# I am not the first person to think or talk about these ideas. The following books are my favorite writers on these topics. While these books focus on building commercial products, most of the ideas apply equally well to open source software.
# 
# * [The Lean Startup](http://theleanstartup.com/) by Eric Ries 
# * [Getting Real](http://gettingreal.37signals.com/) by 37 Signals (free PDF!)
# 
# Here are some IPython specific resources:
# 
# * [IPython Roadmap](https://github.com/ipython/ipython/wiki/Roadmap:-IPython)
# * [IPython GitHub Issues](https://github.com/ipython/ipython/issues?state=open)
# 
# My blog where I have written further about these ideas:
# 
# * http://brianegranger.com

# <markdowncell>

# <h1 class="bigtitle">Thanks!</h1>

# <codecell>

from IPython.display import display, HTML
s = """

<style>

.rendered_html {
    font-family: "proxima-nova", helvetica;
    font-size: 150%;
    line-height: 1.3;
}

.rendered_html h1 {
    margin: 0.25em 0em 0.5em;
    color: #015C9C;
    text-align: center;
    line-height: 1.2; 
    page-break-before: always;
}

.rendered_html h2 {
    margin: 1.1em 0em 0.5em;
    color: #26465D;
    line-height: 1.2;
}

.rendered_html h3 {
    margin: 1.1em 0em 0.5em;
    color: #002845;
    line-height: 1.2;
}

.rendered_html li {
    line-height: 1.5; 
}

.prompt {
    font-size: 120%; 
}

.CodeMirror-lines {
    font-size: 120%; 
}

.output_area {
    font-size: 120%; 
}

#notebook {
    background-image: url('files/images/witewall_3.png');
}

h1.bigtitle {
    margin: 4cm 1cm 4cm 1cm;
    font-size: 300%;
}

h3.point {
    font-size: 200%;
    text-align: center;
    margin: 2em 0em 2em 0em;
    #26465D
}

.logo {
    margin: 20px 0 20px 0;
}

a.anchor-link {
    display: none;
}

h1.title { 
    font-size: 250%;
}

</style>
"""
display(HTML(s))

# <codecell>


