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
    color: darkviolet;
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
    #26465D
}

.logo {
    margin: 20px 0 20px 0;
}

a.anchor-link {
    display: none;
}

h1.title { 
    font-size: 200%;
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

# # Wanted
# 
# ## Comprehensive study of solutions (dynamic behaviors) of a nonlinear dynamical system, the vocal folds in the larynx

# <markdowncell>

# # Wanted
# 
# ## Catalog of complete dynamic behavior of vocal folds as a function of glottal posture

# <markdowncell>

# # Why
# ## Reveal basic mechanisms of neuromuscular control of F0, loudness, voice quality, etc.
# ## Create framework to assess dynamic equivalence of different larynges (human, dog, bats, mammals, birds)
# ## Use systematic experimental data for validation of models

# <markdowncell>

# # Simple Idea: Bifurcation Analysis in an in vivo dog experiment!
# 
# # Comprehensively and systematically!

# <markdowncell>

# # Bifurcation Analysis in In Vivo Dog
# 
# ## Phonation onset (Hopf bifurcation) during air flow ramp
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/bifurcation.png" width=900 />
# </center>

# <markdowncell>

# # Types of bifurcations and nonlinear phenomena
# 
# ## Hopf bifurcation (e.g. phonation onset)
# ## Subharmonic bifurcations; folded limit cycle oscillations
# ## Frequency jumps: chest -- falsetto register transition
# ## Secondary Hopf bifurcation; toroidal oscillation -- biphonation
# ## Bifurcations to chaotic vibrations

# <markdowncell>

# # Neuromuscular Control Parameters
# 
# ## Muscles are actuators: deform larynx, determine the posture
# ## Laryngeal muscles set the tone (posture and stiffness) in terms of strain and stress that the subglottal pressure and flow can play with
# ## Control the actuators via their connected nerves: SLN, RLN, and branches of the RLN (TA, LCA/IA, PCA)
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/postures.png" width=800 />
# </center>

# <markdowncell>

# ToDo:
# 
# show kymogram with data overlays or on top of each other: highlight the POSTURE and the VIBRATION in the high speed video
# 
# 
# show strain data and vocal-process distance
# 
# NO hemilarynx

# <markdowncell>

# # Neuromuscular stimulation scenarios
# 
# ## Left-Right Asymmetric stimulation of muscle groups (e.g. left versus right TA)
# ## Agonist-Antagonist imbalance -- Left-Right Symmetric stimulation of different muscle groups (e.g. SLN versus TA)

# <markdowncell>

# # Example: asymmetric SLN for constant mid RLN
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/asymmetricSLN_MidRLN.pout.specgram.600Hz.png" width=1000 />
# </center>

# <markdowncell>

# # Example: trunk RLN versus TA for constant (no) SLN
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/trunkRLN_TA_NoSLN Wed Mar 21 2012 17 18 17.psub.specgram.1000Hz.mod.png" width=1000 />
# </center>

# <markdowncell>

# # Phonation onset: Phonation frequency (F0) -- vocal fold length (strain)

# <codecell>

F1 = WImage(filename = 'Figure1.pdf')
F1alt = WImage(filename = 'Figure1_alt.pdf')

# <codecell>

display(F1)

# <markdowncell>

# # Detail: Influence of TA activation

# <codecell>

display(F1alt)

# <markdowncell>

# # Example: SLN versus trunk RLN for different TA
# 
# ## High speed video recording: glottal posturing and vocal fold vibration
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/TAatOnset.mod.png" width=1000 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN_trunkRLN_DifferentTA.kymoim.mod.png" width=1000 />
# </center>

# <markdowncell>

# # Example: Repeat Experiments
# ## Left Vagal Paralysis: Right SLN versus Right RLN
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/right SLN versus right RLN baseline no implant Wed Oct 30 2013 16 25 51.pout.specgram.1000Hz.mod.png" width=450 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/right SLN versus right RLN no implant baseline repeat Wed Oct 30 2013 17 47 30.pout.specgram.1000Hz.mod.png" width=450 />
# </center>

# <markdowncell>

# # Repeat Experiments
# 
# ## Left Recurrent Nerve Paralysis: SLN versus right RLN
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN versus right RLN No implant Wed Oct 30 2013 14 52 50.pout.specgram.1000Hz.mod.png" width=450 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/SLN versus right RLN No implant repeat Wed Oct 30 2013 16 05 22.pout.specgram.1000Hz.mod.png" width=450 />
# </center>

# <markdowncell>

# # Use the bifurcation behavior as a metric to compare different dynamical systems:
# 
# ## Different larynges: human, dog, sheep, etc: Are they dynamically equivalent?
# ## Different intervention procedures for voice pathologies: implants, arytenoid adduction, augmentation, mass injection
# ## Different grades of paresis/paralysis (muscle weakness)

# <markdowncell>

# # Goal: Systematic Catalog of dynamical behaviors as a function of posture and stimulation level
# 
# ## Asymmetry
# ## Agonists-Antagonists actions, groups
# ## Redundancies in musculo-skeletal framework
# ## Equivalences in musculo-skeletal framework

# <markdowncell>

# ## to infer INTERNAL laryngeal state from kinematic and dynamic behavior

# <markdowncell>

# # The Not-So-Simple Details: Tight Experimental Control! Fast and Automated Experimental Runs!
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

# # In vivo dog experiment
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/sketch_setup.png" width=700 />
# </center>

# <markdowncell>

# # Software abstraction and automation
# ## Efficient and fast management of experimental complexities
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/dog_experiment_nerve_settings.png" width=1000 />
# </center>

# <markdowncell>

# # Laryngeal Nerve Stimulation
# 
# ## Tripolar Cuff Electrodes, used in Functional Electrical Stimulation (FES)
# ## Stimulation Pulse Trains: short, rectangular, biphasic, charge-balanced (30 microseconds per phase)
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-31/DSC_0005.mod.png" width=400 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 10-17, Dog Experiment CHS/DSC_0008.jpg" width=400 />
# </center>

# <markdowncell>

# # Humidified glottal air flow
# ## Fully humidified and heated subglottal air flow, up to 1600 ml/s
# ## Heated supply lines and subglottal expansion chamber: avoid heat and humidification loss
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-31/DSC_0003.mod.png" width=450 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0052.mod.png" width=450 />
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0056.mod.png" width=500 />
# -->
# </center>

# <markdowncell>

# # Experimental bifurcation parameter
# 
# ## Computer-controlled linear flow ramp -- Increasing subglottal pressure
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-16, Dog Experiment CHS/DSC_0039.mod.png" width=800 />
# </center>

# <markdowncell>

# # Nerve stimulation control (up to 8 nerves)
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

# # Electrode-nerve interface monitoring
# 
# ## Detect current shunting and impedance changes by monitoring injected current and voltage
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/dog_experiment_current_voltage.png" width=1000 />
# </center>

# <markdowncell>

# # Rapid setup of nerve stimulation parameters (up to 8 nerves): 
# 
# ## Binary search for threshold of nerve excitation (one threshold in 10 seconds)
# ## Stimulation range finding assisted by visual of posturing and transglottal pressure drop change
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/2013-10-29 LabView control setup for dog phonation experiments/search_excitation_threshold.png" width=1000 />
# </center>

# <rawcell>

# ## exploit diagnostic tools for range finding: subglottal pressure for constant flow rate
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/" width=500 />
# </center>

# <markdowncell>

# # EMG of Laryngeal Muscles
# 
# ## Excitation threshold finding, Stimulation range finding
# ## Measure relative strength and speed of muscle response (medial, lateral TA, CT, LCA)
# ## Study impact of pulse train parameters: pulse repetition rate, pulse shape
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/EMGtraces_09.mod.png" width=600 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0011.mod.png" width=350 />
# 
# <!--
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 11-14 Dog Experiment CHS/DSC_0014.mod.png" width=450 />
# -->
# 
# </center>

# <markdowncell>

# # Need for Speed: A recording every 5 seconds
# 
# ## High speed motion capture includes prephonatory posturing and vocal fold vibration
# ## 5 minutes to record a comprehensive, systematic set of 64 stimulation conditions
# ## Stimulation range finding and checks also recorded: single nerve stimulation ramps
# 
# <center>
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 02-22, Dog experiment CHS/DSC_0008.mod.png" width=800 />
# <img src="files/ICVPB2014_Salt_Lake_City.images/Lab UCLA 2012 03-14, Dog Experiment CHS/DSC_0005.mod.png" width=800 />
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
# 
# ## Parameterize vocal posture via laryngeal nerve stimulation
# 
# ## Perform fast setup and experimental runs for consistent dynamic behavior
# 
# ## Monitor controls in realtime and offline
# 
# ## Repeat experiments show comparable robust bifurcation scenarios
# 
# ## Bifurcations due to: Left-Right Asymmetries; Agonist-Antagonist Muscle Imbalance

