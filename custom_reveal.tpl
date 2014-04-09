{%- extends 'slides_reveal.tpl' -%}

{% block body %}

{{ super() }}

<script>

Reveal.initialize({

    // Display controls in the bottom right corner
    //controls: true,

    // Display a presentation progress bar
    //progress: true,

    // Push each slide change to the browser history
    //history: false,

    // Enable keyboard shortcuts for navigation
    //keyboard: true,

    // Enable touch events for navigation
    //touch: true,

    // Enable the slide overview mode
    //overview: true,

    // Vertical centering of slides
    //center: true,

    // Loop the presentation
    //loop: false,

    // Change the presentation direction to be RTL
    //rtl: false,

    // Number of milliseconds between automatically proceeding to the
    // next slide, disabled when set to 0, this value can be overwritten
    // by using a data-autoslide attribute on your slides
    //autoSlide: 0,

    // Enable slide navigation via mouse wheel
    //mouseWheel: false,

    // Transition style
    // transition: 'concave', // default/cube/page/concave/zoom/linear/fade/none

    // Transition speed
    //transitionSpeed: 'default', // default/fast/slow

    // Transition style for full page backgrounds
    //backgroundTransition: 'default', // default/linear/none

    // Theme
    // theme: 'sky', // available themes are in /css/theme

controls: true,
progress: true,
history: true,

slideNumber: Reveal.getQueryHash().slideNumber || true,
touch: false,

// themes: default, beige, sky, night, serif, simple, solarized, none
theme: Reveal.getQueryHash().theme || 'serif', // available themes are in /css/theme

transition: Reveal.getQueryHash().transition || 'none', // default/cube/page/concave/zoom/linear/none

width: Reveal.getQueryHash().width || 1500, // default: 960
height: Reveal.getQueryHash().height || 1000, // 1200, // default: 700

// factor of display size that should remain empty
margin: Reveal.getQueryHash().margin || 0.0, // default: 0.1

// bounds for smallest/largest possible scale to apply to content
minScale: 0.2, // default: 0.2
maxScale: 1.0, // default: 1.0

});

</script>

{% endblock body %}
