# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import gst, gtk, gobject

# <codecell>

class VideoPlayer:
    def __init__(self):
        self.window = gtk.Window()
        self.window.connect('destroy', self.on_destroy)

        self.drawingarea = gtk.DrawingArea()
        self.drawingarea.connect('realize', self.on_drawingarea_realized)
        self.window.add(self.drawingarea)

        self.playbin = gst.element_factory_make('playbin2')
        self.playbin.set_property('uri', 'file:///home/neubauer/Downloads/water-and-wind.ogv')

        self.sink = gst.element_factory_make('xvimagesink')
        self.sink.set_property('force-aspect-ratio', True)
        self.playbin.set_property('video-sink', self.sink)

        self.bus = self.playbin.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::eos", self.on_finish)

        self.window.show_all()

        self.playbin.set_state(gst.STATE_PLAYING)

    def on_finish(self, bus, message):
        self.playbin.set_state(gst.STATE_PAUSED)

    def on_destroy(self, window):
        self.playbin.set_state(gst.STATE_NULL)
        gtk.main_quit()

    def on_drawingarea_realized(self, sender):
        self.sink.set_xwindow_id(self.drawingarea.window.xid)

# <codecell>

VideoPlayer()
gtk.main()

# <codecell>

gst.get_gst_version

# <codecell>

gst.get_pygst_version

# <codecell>

gst.version

# <codecell>

gst.version_string

# <codecell>


