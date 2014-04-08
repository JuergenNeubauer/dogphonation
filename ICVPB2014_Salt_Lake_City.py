# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <rawcell>

# <style type="text/css">
#     .input, .output_prompt {
#             display:none !important;
#     }
#     
#     table, th, tr, td {
#             border: none;
#             text-align: center;
#             align-items: center;
#             align-content: center;
#             background-color: orange;
#     }
#     
#     figure {
#         border: none;
#         text-align: center;
#         display: inline;
#     }
#     
#     figcaption {
#         border: none;
#         text-align: center;
#     }
# </style>

# <codecell>

from IPython.display import display, HTML, Image
from wand.image import Image as WImage

# <codecell>

s = """

<style>

div.cell, div.text_cell_render {
        width:100%;
        margin-left:1%;
        margin-right:auto;
}

.rendered_html {
    font-family: "proxima-nova", helvetica;
    font-size: 150%;
    line-height: 1.3;
}

.rendered_html h1 {
    margin: 0.25em 0em 0.5em;
    # color: #015C9C;
    color: #CC3300;
    text-align: center;
    line-height: 1.2; 
    page-break-before: always;
    font-size: 250%;
}

.rendered_html h2 {
    margin: 1.1em 0em 0.5em;
    color: black;
    line-height: 1.2;
    text-align: center;
}

.rendered_html h3 {
    margin: 1.1em 0em 0.5em;
    color: black;
    line-height: 1.2;
}

.rendered_html li {
    font-size: 120%
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
    color: #26465D;
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

# div.cell{
#         max-width:750px;
#         margin-left:auto;
#         margin-right:auto;
# }


</style>
"""

display(HTML(s))

# <markdowncell>

# <center>
# <h1 class='title'> Comprehensive Bifurcation Analysis<br>in a Neuromuscularly-Controlled <br>In Vivo Canine Larynx </h1>
# 
# <h2 > Juergen Neubauer and Dinesh K. Chhetri </h2>
# 
# <h3 > Simon Levin MCMSC @ ASU <br> Head and Neck Surgery @ UCLA </h3>
# 
# <h4 > ICVPB Salt Lake City, 2014 </h4>
# 
# <h4> Supported by NIH RO1 DC011300</h4>
# </center>

# <rawcell>

# Over the last couple of years we developed an experimental setup to catalog the complete vocal fold dynamics in our in vivo dog model.

# <markdowncell>

# # Wanted
# 
# ## Complete description of the vocal fold dynamics in an in vivo model

# <rawcell>

# So why is this necessary? We need such a complete description to understand the neuromuscular control of fundamental frequency, loudness, voice quality and pathologies. We need it to assess the dynamic equivalence of larynges of different species. And we need it to validate theoretical and computational models.

# <markdowncell>

# # Why
# ## Neuro-muscular control of F0, loudness, voice quality, pathologies
# ## Dynamic equivalence of larynges of different species
# ## Model validation

# <rawcell>

# How are we doing this? We do this with a comprehensive and systematic experimental bifurcation analysis. 
# 
# For example, I'm showing the most interesting bifurcation, the phonation onset, which is a Hopf bifurcation. 
# 
# In this spectrogram of the subglottal acoustic signal, there is a transition from aphonia to vocal fold vibration as demonstrated by the appearing stack of harmonics. 
# 
# This happens due to an increase in subglottal pressure.

# <markdowncell>

# # Bifurcation analysis in *in vivo* dog experiment
# 
# ## Phonation onset, a Hopf bifurcation during air flow ramp
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/bifurcation.png" width=900 />
# </center>

# <rawcell>

# The phonation onset is only one example of bifurcations that we typcially see in our experiments. 
# 
# We also all other kinds of bifurcations that typcially occur in nonlinear systems. 
# 
# Interestingly, we observe frequency jumps that might be linked to chest-falsetto register transitions.

# <markdowncell>

# # Observed bifurcations
# 
# ## Hopf bifurcation
# ## Subharmonic bifurcation
# ## Frequency jumps
# ## Secondary Hopf bifurcation
# ## Sudden transition to chaotic vibrations

# <rawcell>

# When do we get these bifurcations? Different bifurcations appear for different kinds of manipulations of the laryngeal muscles. Muscles are like actuators, they deform the larynx, and so they set the vocal fold posture and the internal stiffnesses. When we blow air through such a deformed larynx, these bifurcations appear.
# 
# In our experiments we control the muscles by stimulating the nerves. These are the SLN, RLN, and branches of the RLN.
# 
# This example shows all locations of landmark points on the vocal folds for different muscle activations.

# <markdowncell>

# # Bifurcations induced by neuromuscular control
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/postures.png" width=800 />
# </center>

# <markdowncell>

# # Bifurcations induced by air flow ramp
# 
# ## Computer-controlled linear flow ramp -- Increasing subglottal pressure
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0039.mod.png" width=800 />
# </center>

# <rawcell>

# # Neuromuscular stimulation scenarios
# 
# ## Left-Right Asymmetric stimulation of muscle groups (e.g. left versus right TA)
# ## Agonist-Antagonist imbalance -- Left-Right Symmetric stimulation of different muscle groups (e.g. SLN versus TA)

# <rawcell>

# There are a couple of particularly interesting manipulations of the larynx. First I'll show an example of a left-right asymmetric stimulation experiment. 
# 
# Such a manipulation simulates a unilateral paresis and paralysis pathology.
# 
# The spectral bifurcation diagrams look similar on both sides of the diagonal. 
# 
# For low levels of SLN stimulation, we can see more regular phonation. For medium levels, broadband and chaotic oscillations appear with transitions to regular phonation. For the highest levels, again more regular phonation, now with higher fundamental frequency appear.

# <markdowncell>

# # Left-right asymmetric stimulation
# 
# 
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/asymmetricSLN_MidRLN.pout.specgram.600Hz.png" width=1000 />
# </center>
# 
# ## asymmetric SLN for constant mid RLN

# <markdowncell>

# # Agonist-antagonist imbalance
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/trunkRLN_TA_NoSLN Wed Mar 21 2012 17 18 17.psub.specgram.1000Hz.mod.png" width=1000 />
# </center>
# 
# ## trunk RLN versus TA for constant (no) SLN

# <markdowncell>

# # Phonation onset: Phonation frequency (F0) -- vocal fold length (strain)
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Figure1.png" width=600 />
# </center>

# <markdowncell>

# # Tight experimental control. Fast and automated experiments.
# 
# <table>
# <tr>
# <td>
# <figure>
#     <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 03-14, Dog Experiment CHS/DSC_0008.mod.png" width=600 />
#     <figcaption>Controller</figcaption>
# </figure>
# </td>
# <td>
# <figure>
#     <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0014.mod.png" width=600 />
#     <figcaption>In vivo dog model</figcaption>
# </figure>
# </td>
# <td>
# <figure>
#     <img src="files/ICVPB2014_Salt_Lake_City.images/cables.mod.png" width=600 />
#     <figcaption>Extensive control and<br> recording infrastructure</figcaption>
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-22, Dog experiment CHS/DSC_0001.jpg" width=300 />
# -->
# </figure>
# </td>
# </tr>
# </table>

# <markdowncell>

# # Nerve stimulation control (8 nerves)
# 
# ## Computer-controlled and automated stimulation pulse train sequences
# 
# <center>
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0004.jpg" width=500 />
# -->
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0009.mod.png" width=700 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/dog_experiment_nerve_settings.png" width=700 />
# </center>

# <markdowncell>

# # Fast experiments: A recording every 5 seconds
# 
# ## High speed motion capture includes prephonatory posturing and vocal fold vibration
# ## 5 minutes to record a comprehensive, systematic set of 64 stimulation conditions
# ## Stimulation range finding and checks also recorded: single nerve stimulation ramps
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-22, Dog experiment CHS/DSC_0008.mod.png" width=800 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 03-14, Dog Experiment CHS/DSC_0005.mod.png" width=800 />
# </center>

# <markdowncell>

# # Rapid setup of nerve stimulation parameters: 
# 
# ## Binary search for threshold of nerve excitation (one threshold in 10 seconds)
# ## Stimulation range finding assisted by visual of posturing and transglottal pressure drop change
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/search_excitation_threshold.png" width=1000 />
# </center>

# <markdowncell>

# # Replication possible!
# 
# ## First take --- Second take
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN versus right RLN No implant Wed Oct 30 2013 14 52 50.pout.specgram.1000Hz.mod.png" width=450 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN versus right RLN No implant repeat Wed Oct 30 2013 16 05 22.pout.specgram.1000Hz.mod.png" width=450 />
# </center>
# 
# ## Left Recurrent Nerve Paralysis: SLN versus right RLN

# <markdowncell>

# # Consistent data from in vivo experiment possible!
# 
# # Bifurcation behavior: Metric to compare dynamical systems:
# 
# ## Larynges: human, dog, sheep, etc: Are they dynamically equivalent?
# ## Intervention procedures for voice pathologies: implants, arytenoid adduction, injection
# ## Grades of paresis/paralysis

# <rawcell>

# # Goal: Systematic catalog of dynamical behaviors as a function of posture and stimulation level
# 
# ## Asymmetry
# ## Agonist-Antagonist interactions
# ## Redundancies in laryngeal musculo-skeletal framework
# ## Equivalences in laryngeal musculo-skeletal framework

