# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# [Index](index.ipynb)
# 
# To use IPython widgets in the notebook, the widget namespace needs to be imported.

# <codecell>

from IPython.html import widgets # Widget definitions
from IPython.display import display # Used to display widgets in the notebook

# <headingcell level=1>

# Basic Widgets

# <markdowncell>

# IPython comes with basic widgets that represent common interactive controls.  These widgets are
# 
# - CheckboxWidget
# - ToggleButtonWidget
# - FloatSliderWidget
# - BoundedFloatTextWidget
# - FloatProgressWidget
# - FloatTextWidget
# - ImageWidget
# - IntSliderWidget
# - BoundedIntTextWidget
# - IntProgressWidget
# - IntTextWidget
# - ToggleButtonsWidget
# - RadioButtonsWidget
# - DropdownWidget
# - SelectWidget
# - HTMLWidget
# - LatexWidget
# - TextareaWidget
# - TextWidget
# - ButtonWidget
# 
# A few special widgets are also included, that can be used to capture events and change how other widgets are displayed.  These widgets are
# 
# - ContainerWidget
# - PopupWidget
# - AccordionWidget
# - TabWidget
# 
# To see the complete list of widgets, one can execute the following

# <codecell>

[widget for widget in dir(widgets) if widget.endswith('Widget')]

# <markdowncell>

# The basic widgets all have sensible default values.  Create a *FloatSliderWidget* without displaying it:

# <codecell>

mywidget = widgets.FloatSliderWidget()

# <markdowncell>

# Constructing a widget does not display it on the page.  To display a widget, the widget must be passed to the IPython `display(object)` method or must be returned as the last item in the cell.  `mywidget` is displayed by

# <codecell>

display(mywidget)

# <markdowncell>

# or

# <codecell>

mywidget

# <markdowncell>

# It's important to realize that widgets are not the same as output, even though they are displayed with `display`.  Widgets are drawn in a special widget area.  That area is marked with a close button which allows you to collapse the widgets.  Widgets cannot be interleaved with output.  Doing so would break the ability to make simple animations using `clear_output`.
# 
# Widgets are manipulated via special instance attributes (traitlets).  The names of these traitlets are listed in the widget's `keys` attribute (as seen below).  A few of these attributes are common to most widgets.  The basic attributes are `value`, `description`, `visible`, and `disabled`.  `_css` and `_view_name` are private attributes that exist in all widgets and should not be modified.

# <codecell>

mywidget.keys

# <markdowncell>

# Changing a widget's attribute will automatically update that widget everywhere it is displayed in the notebook.  Here, the `value` attribute of `mywidget` is set.  The slider shown above updates automatically with the new value.  Syncing also works in the other direction - changing the value of the displayed widget will update the property's value.

# <codecell>

mywidget.value = 25.0

# <markdowncell>

# After changing the widget's value in the notebook by hand to 0.0 (sliding the bar to the far left).

# <codecell>

mywidget.value

# <markdowncell>

# Widget values can also be set with kwargs during the construction of the widget (as seen below).

# <codecell>

mysecondwidget = widgets.RadioButtonsWidget(values=["Item A", "Item B", "Item C"], value="Item A")
display(mysecondwidget)

# <codecell>

mysecondwidget.value

# <markdowncell>

# In [Part 2](Part 2 - Events.ipynb) of this [series](index.ipynb), you will learn about widget events.

