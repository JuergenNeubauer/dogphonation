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
# <h4> Supported by NIH RO1 DC011300 </h4>
# </center>

# <markdowncell>

# # Wanted
# 
# ## Complete description of the vocal fold dynamics in an *in vivo* model

# <rawcell>

# I'm going to talk about our in vivo dog experiments and how we use bifurcation analysis
# 
# We have been working on a method in the last couple of years to measure, record, and catalog the complete vocal fold dynamics in an in vivo dog experiment.

# <markdowncell>

# # Why
# ## Neuro-muscular control of F0, loudness, voice quality, pathologies
# ## Dynamic equivalence of larynges of different species
# ## Model validation

# <rawcell>

# So why is this necessary? 
# 
# We need such a complete description to understand fundamental mechanisms of the neuromuscular control of fundamental frequency, control of loudness, voice quality and control of pathological voice production. 
# 
# We need it to assess the dynamic equivalence of larynges of different species. 
# 
# We also need it to validate theoretical and computational models.

# <markdowncell>

# # Bifurcation analysis in *in vivo* dog experiment
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/bifurcation.png" width=900 />
# </center>
# 
# ## Phonation onset, a Hopf bifurcation during air flow ramp

# <rawcell>

# How are we actually doing this? 
# 
# We apply experimental bifurcation analysis for a comprehensive and systematic set of laryngeal conditions.
# 
# For example, here I'm showing a experimental spectrogram which demonstrates the most important bifurcation, the phonation onset, a Hopf bifurcation. 
# 
# In this spectrogram of the subglottal acoustic signal, a transition occurs from aphonia to vocal fold vibration. This transition happens where the stack of harmonics appear on the right side of the spectrogram. 
# 
# Here, I used an increasing subglottal pressure as the bifurcation parameter. So phonation onset happened when the subglottal pressure was sufficiently high.
# 
# I will call this kind of spectrogram 'spectral bifurcation diagram'. 
# 
# I will use these spectral bifurcation diagrams as the basic building blocks to characterize the vocal fold dynamics of all kinds of laryngeal conditions.

# <markdowncell>

# # Observed bifurcations
# 
# ## Hopf bifurcation
# ## Subharmonic bifurcation
# ## Frequency jumps
# ## Secondary Hopf bifurcation
# ## Sudden transition to chaotic vibrations

# <rawcell>

# The phonation onset is only one example of bifurcations that we typcially see in our experiments. 
# 
# We also observe all other kinds of bifurcations that typically occur in nonlinear systems. 

# <markdowncell>

# # Bifurcations induced by neuromuscular control
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/postures.png" width=800 />
# </center>
# 
# ## Computer-controlled, automated pulse train sequences (1500 ms)

# <rawcell>

# Different bifurcations appear for different kinds of manipulations of the laryngeal muscles. 
# 
# Muscles are like actuators, they deform the larynx. So they set the vocal fold posture and the internal stiffnesses. 
# 
# When we blow air through such a deformed larynx, all kinds of bifurcations can appear.
# 
# In our experiments we control the muscles by stimulating their nerves. 
# 
# These are the SLN, RLN, and branches of the RLN.
# 
# This example shows all locations of landmark points on the vocal folds for different muscle activations.

# <markdowncell>

# # Bifurcations induced by air flow ramp
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0039.mod.png" width=800 />
# </center>
# 
# ## Computer-controlled linear flow ramp -- Increasing subglottal pressure

# <rawcell>

# Bifurcations also occur when we apply an air flow ramp to each of the muscle manipulations. 
# 
# Due to some glottal flow resistance, this causes the subglottal pressure to increase. 
# 
# Then, usually, phonation onset occurs, or, for even high pressures, all other kinds of bifurcations can occur.

# <markdowncell>

# # Left-right asymmetric stimulation
# 
# 
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/asymmetricSLN_MidRLN.pout.specgram.600Hz.mod.png" width=1000 />
# </center>
# 
# ## right SLN versus left SLN, constant mid RLN

# <rawcell>

# There are a two particularly interesting manipulations of the larynx. 
# 
# First I'll show an example of a left-right asymmetric stimulation experiment. 
# 
# This manipulation simulates a unilateral paresis or a unilateral vocal fold paralysis.
# 
# Here I show a whole array of spectral bifurcation diagrams: 
# Along each row, the left SLN level increases in 7 steps. 
# Along each column, the right SLN level increases also in 7 steps.
# 
# The overall array of spectral bifurcation diagrams looks pretty symmetric with respect to the diagonal line. That what we would expect if the true activation of the muscles was symmetric and the larynx did not have any intrinsic tissue asymmetries.

# <markdowncell>

# # Agonist-antagonist imbalance
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/trunkRLN_TA_NoSLN Wed Mar 21 2012 17 18 17.psub.specgram.1000Hz.mod.png" width=1000 />
# </center>
# 
# ## TA versus trunk RLN (LCA/IA), constant No SLN

# <rawcell>

# The second case is an example for an agonist-antagonist imbalance. Here the left and right side were stimulated symmetrically, but different TA levels were paired with different trunk RLN levels.
# 
# Here, trunk RLN stimulation levels increasing along the rows and TA levels increase along the columns.
# 
# This kind of muscle manipulation is of fundamental importance to study the influence of vocal fold posture on phonation. What we want to know is how muscle synergies create elemental postures and how these elemental postures  are linked to the different kinds of bifurcations.
# 
# In the shown array of spectral bifurcation diagrams I want to point out two features. One is the appearance of more broadband chaotic vibrations for high TA and high trunk RLN levels, shown in the upper right corner. The other is the fact that TA activation decreases the fundamental frequency at onset.

# <markdowncell>

# # Phonation onset for SLN -- TA -- trunk RLN manipulation
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Figure1.png" width=450 />
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Figure1_alt.png" width=450 />
# -->
# </center>
# 
# ## chest and falsetto-like clusters at phonation onset

# <rawcell>

# Now when we analyzed the phonation onset behavior for the entire set of TA, trunk RLN and SLN manipulations, we found the following:
# 
# At phonation onset, we measure the onset frequency, subglottal pressure and flow, and we measure vocal fold strain and adduction from the high speed videos of the larynx.
# 
# Here, I show the very surprising finding that onset frequencies cluster in two branches as a function of vocal fold strain. It's surprising because it's reminiscent of chest and falsetto registers at phonation onset.

# <markdowncell>

# # Tight experimental control. Fast and automated setup and experiments.
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

# <rawcell>

# The data that I've shown so far was only possible to get with a well-controlled experiment.
# 
# We have spent a lot of effort building an in vivo dog experiment with tight experimental controls, that we can execute quickly and that runs experiments automatically.

# <markdowncell>

# # Computer-controlled and automated nerve stimulation pulse train sequences (8 nerves)
# 
# <center>
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0004.jpg" width=500 />
# -->
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0009.mod.png" width=700 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/dog_experiment_nerve_settings.png" width=700 />
# </center>

# <rawcell>

# I have shown already that the air flow ramp is automatically generated and controlled by a computer.
# 
# Also the stimulation pulse trains that we apply to the nerves are fully computer-controlled and the level selection is fully automated.

# <markdowncell>

# # Rapid setup
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/search_excitation_threshold.png" width=1000 />
# </center>
# 
# ## Binary search for threshold of nerve excitation (one threshold in 10 seconds)

# <rawcell>

# We can get the in vivo experiments set up quickly.
# 
# For example, we use a binary search to determine the stimulation thresholds of the nerves very quickly.

# <markdowncell>

# # Fast experiments: New recording every 5 seconds
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-22, Dog experiment CHS/DSC_0008.mod.png" width=800 />
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 03-14, Dog Experiment CHS/DSC_0005.mod.png" width=800 />
# -->
# </center>
# 
# ## High speed video: prephonatory posturing, vocal fold vibration
# ## 5:20 min for 64 stimulation conditions

# <rawcell>

# And we measure relevant variables every five seconds.
# 
# That includes subglottal pressure and flow rate, subglottal acoustics, outside acoustics, EMG from laryngeal muscles, the electrode-nerve impedances, and, last but not least,  high speed video data of the prephonatory posturing behavior and vocal fold vibration.
# 
# So for example in 5 minutes and 20 seconds we can record a complete set of 8 times 8 , so 64, different stimulation conditions.

# <markdowncell>

# # Replication possible!
# 
# ## First take ---- Second take
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN versus right RLN No implant Wed Oct 30 2013 14 52 50.pout.specgram.1000Hz.mod.png" width=550 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN versus right RLN No implant repeat Wed Oct 30 2013 16 05 22.pout.specgram.1000Hz.mod.png" width=550 />
# </center>
# 
# ## Left recurrent nerve paralysis: both SLN versus right RLN

# <rawcell>

# As a result, we are now able to replicate bifurcation behaviors when we repeat experiments in the same dog.
# 
# What I show here is a repeat experiment of a paralysis condition in the same dog. You can see that the spectral bifurcatoin diagrams are similar for the first and the second take of this repeat experiment.
# 
# So now we are able to use this bifurcation behavior, symbolized by the spectral bifurcation diagrams, to compare and quantify the overall dynamic behavior, either in the same dog, or between dogs, or across species, and so on.

# <markdowncell>

# # Complete bifurcation behavior provides a metric to compare dynamical systems
# 
# ## Metric to measure dynamical equivalence of different larynes: human, dog, bats, etc
# ## Metric to evaluate intervention procedures for voice pathologies: implants, arytenoid adduction, injection

# <rawcell>

# What this means is that we can now collect consistent dynamic data from our in vivo dog experiment with high confidence and at high speed. That allows us to systematically catalog the complete bifurcation behavior.
# 
# So now it makes sense to use these sets of bifurcation behaviors as a metric to compare different dynamical systems.
# 
# Different dynamical systems could be different dogs, or different species (either human or dog or bats). So we can now quantitatively answer the question how dynamically equivalent the vocal folds of these different species really are.
# 
# And last, we can use this bifurcation metric also to decide how effective different intervention procedures are to correct voice pathologies by using vocal fold implants or arytenoid adduction or mass injection.

