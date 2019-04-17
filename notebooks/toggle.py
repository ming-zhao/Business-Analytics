import IPython.core.display as di
from IPython.display import HTML

di.display_html('''<script>
    function code_toggle(e) {
		while (!e.closest(".nboutput").previousElementSibling.classList.contains("nbinput")) {
			e = e.closest(".nboutput").previousElementSibling;
		}
		var target = e.closest(".nboutput").previousElementSibling;
        if (target.getAttribute("style") == "" || target.getAttribute("style") == null) {
            target.style.display = "none";
        }
        else {
            target.style.removeProperty("display");
        }
    }
	$(document).ready(function() {
		var code_blocks = document.getElementsByClassName('nbinput docutils container');
		for (var i = 0, len = code_blocks.length; i < len; ++i) {
			if (code_blocks[i].innerHTML.indexOf("toggle") !== -1) {
				code_blocks[i].style.display = "none";
			}
		}
		//$('div.nbinput').hide();
	});
    $(document).ready(code_toggle);
</script>''', raw=True)

def toggle():
    html = """
        <a href="#" onclick="code_toggle(this); return false;">show/hide code</a>
    """
    return HTML(html)