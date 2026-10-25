{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

{% if modules %}
.. rubric:: Модули

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
