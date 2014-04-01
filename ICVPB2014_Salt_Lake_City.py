# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# <style type="text/css">
# .input, .output_prompt {
# display:none !important;
# }
# </style>

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
#     background-image: url('files/images/witewall_3.png');
# }

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

# display(HTML(s))

# <markdowncell>

# <center>
# <h1 > Comprehensive Bifurcation Analysis in a <br>Neuromuscularly-Controlled In Vivo Canine Larynx <h1>
# 
# <h2 > Juergen Neubauer and Dinesh K. Chhetri </h2>
# 
# <h3 > Simon Levin MCMSC @ ASU <br> Head and Neck Surgery @ UCLA </h3>
# 
# <h4 > ICVPB Salt Lake City, 2014 </h4>
# 
# <h4> Supported by NIH R01 xyz</h4>
# </center>

# <markdowncell>

# 
# ## Abstract
# 
# > In nonlinear dynamical systems, such as the vocal folds in the larynx,
# bifurcation analysis reveals the complete dynamic behavior. 
# 
# > We present a novel method for performing experimental bifurcation analysis in
# an in vivo model of phonation. Leveraging systematic and software-controlled nerve
# stimulation of up to eight laryngeal nerve groups, subglottal
# flow ramp generation, and synchronous recording of laryngeal EMG, transglottal
# pressure, acoustic pressure signals, and high speed video we measure a
# complete set of bifurcation behaviors in short time. Every five seconds the
# bifurcations of a different glottal posture configuration can be explored. 
# 
# > We demonstrate different neuromuscular stimulation scenarios: left-right asymmetric stimulation of muscle groups; left-right symmetric stimulation of different muscle groups; hemilarynx stimulation.
# 
# > As a function of neuromuscular control we present different kinds of bifurcations: Hopf bifurcation, cascades of subharmonic bifurcations, frequency jumps, and bifurctions to chaotic vibrations.

# <markdowncell>

# # Simple Idea: Do Bifurcation Analysis in an in vivo dog experiment!

# <markdowncell>

# # Not so modest idea: Let's do it comprehensively and systematically!

# <markdowncell>

# # Why Bifurcation Analysis? And what's this all about? Give me an example!

# <markdowncell>

# # How does bifurcation analysis apply to an in vivo dog experiments? Who bifurcates and what are our control parameters?
# 
# ## muscles are actuators: deform larynx, determine the posture
# ## laryngeal muscles set the tone (posture and stiffness) in terms of strain and stress that the subglottal pressure and flow can play with
# ## We control the actuators (sort of) via the nerves connected to them: SLN, RLN, and branches of the RLN: TA, LCA/IA, PCA
# 
# # Demonstrate different neuromuscular stimulation scenarios: 
# 
# * left-right asymmetric stimulation of muscle groups
# * left-right symmetric stimulation of different muscle groups
# * hemilarynx stimulation.

# <markdowncell>

# # What do we get from it? What can we do with it? How does it look like?
# 
# ## Different kinds of bifurcations and nonlinear phenomena: 
# 
# - Hopf bifurcation (e.g. phonation onset)
# - cascades of subharmonic bifurcations; folded limit cycle oscillations
# - frequency jumps (chest - falsetto); secondary Hopf bifurcations
# - bifurcations to chaotic vibrations
# 
# ## Idea: Use the bifurcation behavior as a dynamical metric to compare:
# 
# * different larynges: human, dog, sheep, etc: Are they dynamically equivalent?
# * different intervention procedures for voice pathologies: implants, arytenoid adduction, augmentation, mass injection
# * severity of paresis (muscle weakness) in terms of dynamical effects
# 
# ## Eventually: catalogue of dynamical behaviors to infer INTERNAL laryngeal state from kinematic and dynamic behavior

# <markdowncell>

# # The Not-So-Simple Details: Tight Experimental Control!
# 
# <center>
# <div>
# <h2> Controller
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 03-14, Dog Experiment CHS/DSC_0008.jpg" width=400 />
# </h2>
# <div>
# <h2> In vivo dog model
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0014.jpg" width=400 />
# </h2>
# <div>
# <h2> Extensive control and <br> recording infrastructure
# <img src="files/ICVPB2014_Salt_Lake_City.images/cables.jpg" width=300 />
# </h2>
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-22, Dog experiment CHS/DSC_0001.jpg" width=300 />
# -->
# </center>

# <markdowncell>

# # Tight Experimental Control: Software abstraction enables management of experimental complexities
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/dog_experiment_nerve_settings.png" width=900 />
# </center>

# <markdowncell>

# # Tight Experimental Control:
# 
# ## nerve stimulation: tripolar cuff electrodes, inspired by Functional Electrical Stimulation (FES) community
# 
# ## stimulation pulse train: biphasic, charge-balanced, short-duration (30 microseconds per phase), rectangular
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-31/DSC_0005.jpg" width=500 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 10-17, Dog Experiment CHS/DSC_0008.jpg" width=500 />
# </center>

# <markdowncell>

# # Experimental Control:
# 
# ## fully humidified and heated subglottal air flow, up to 1600 ml/s
# ## no heat and humidification loss in heated supply hoses and subglottal expansion chamber
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-31/DSC_0003.jpg" width=500 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0052.jpg" width=500 />
# </center>

# <markdowncell>

# ## primary experimental bifurcation parameter: air flow control
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0039.jpg" width=500 />
# </center>

# <markdowncell>

# ## stimulation control
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0004.jpg" width=500 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0009.jpg" width=500 />
# </center>

# <markdowncell>

# ## stimulation current and voltage monitoring
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/dog_experiment_current_voltage.png" width=700 />
# </center>

# <markdowncell>

# ## quick setup of controls: 
# 
# # binary search for threshold of nerve excitation
# # stimulation range finding verified by visual of posturing and transglottal pressure drop change
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/search_excitation_threshold.png" width=700 />
# </center>

# <markdowncell>

# ## exploit diagnostic tools for range finding: subglottal pressure for constant flow rate
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/" width=500 />
# </center>

# <markdowncell>

# ## EMG recordings for excitation threshold finding, range finding, and to answer questions about relative strength and speed of muscle response
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0014.jpg" width=500 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0011.jpg" width=500 />
# </center>

# <markdowncell>

# # Need for Speed: A new recording every 5 seconds
# 
# ## entire set of 64 stimulation conditions recorded in 5 minutes and 20 seconds!
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-22, Dog experiment CHS/DSC_0008.jpg" width=500 />
# </center>

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt
_ = plt.plot([1,2,3])

# <markdowncell>

# # Summary
# 
# ## Collect systematic data on kinematic and dynamic behavior of neuro-muscular model of a mammalian larynx
# ## Parameterize vocal posture via laryngeal nerve stimulation
# ## Perform fast setup and experimental runs for consistent dynamic behavior
# ## Monitor controls in realtime and offline
# 
# ## Repeat experiments show comparable bifurcation scenarios
# ## Bifurcations induced by asymmetric stimulation conditions or imbalance of agonist-antagonist muscle actions

# <codecell>


