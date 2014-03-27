# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# <center>
# <h1 > Comprehensive Bifurcation Analysis in a <br>Neuromuscularly-Controlled In Vivo Canine Larynx <h1>
# 
# <h2 > Juergen Neubauer and Dinesh K. Chhetri </h2>
# 
# <h3 > Simon Levine MCMSC @ ASU, Head and Neck Surgery @ UCLA
# 
# <h3 > ICVPB Salt Lake City, 2014 </h3>
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

# # How does that relate to the in vivo dog experiments? Who bifurcates and what are your control variables?
# 
# * muscles are actuators: deform larynx, determine the posture
# * laryngeal muscles set the tone (aka posture) in terms of strain and stress that the subglottal pressure and flow can play with
# * We control the actuators (sort of) via the nerves connected to them: SLN, RLN, and branches of the RLN: TA, LCA/IA, PCA

# <markdowncell>

# # What do we get from it? What can we do with it? How does it look like?

# <markdowncell>

# # The Not-So-Simple Details: Tight Experimental Control!

# <markdowncell>

# * nerve stimulation: electrode choice

# <markdowncell>

# * humidification and heated air flow

# <markdowncell>

# * air flow control
# * stimulation control
# * stimulation current and voltage monitoring
# * quick setup: threshold of nerve excitation and stimulation range finding
# * exploit diagnostic tools for range finding: subglottal pressure for constant flow rate
# * EMG recordings for excitation threshold finding, range finding, and to answer questions about relative strength and speed of muscle response

# <markdowncell>

# # Need for Speed: A new recording every 5 seconds
# 
# * entire set of 64 stimulation conditions recorded in 5 minutes and 20 seconds!

# <codecell>

%matplotlib inline

# <codecell>

import matplotlib.pyplot as plt
plt.plot([1,2,3])

# <markdowncell>

# # Summary
# 
# * Collect systematic data on kinematic and dynamic behavior of neuro-muscular model of a mammalian larynx

# <codecell>


