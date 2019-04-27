import IPython.core.display as di
from IPython.display import HTML

di.display_html('''<script>
  var password,
      teacher_mode,
      isHtml;
      
  var class_output,
      class_input,
      class_answer;
      
  function code_toggle(e) {
    while (!e.closest(class_output).previousElementSibling.classList.contains(class_input)) {
      e = e.closest(class_output).previousElementSibling;
    }
    var target = e.closest(class_output).previousElementSibling;
    if (target.getAttribute("style") == "" || target.getAttribute("style") == null) {
      target.style.display = "none";
      e.innerHTML = 'show code';
    }
    else {
      target.style.removeProperty("display");
      e.innerHTML = 'hide code';
    }
  }
  
  function hide_answer(e) {
    var target = e.closest(class_answer).nextElementSibling;
    //e.closest(class_output).previousElementSibling.style.display = "none";
    if (target.getAttribute("style") == "" || target.getAttribute("style") == null) {
      target.style.display = "none";
      e.innerHTML = 'show answer';
    }
    else if (teacher_mode) {
        e.innerHTML = 'hide answer';
        target.style.removeProperty("display");
    }
  }
  
  function done() { 
    document.getElementById("popup").style.display = "none";
    var input = document.getElementById("password").value;
    if (input==password) { teacher_mode=1; alert("Unlocked!");}
    else { teacher_mode=0; alert("Wrong password!");}       
  };

  function unlock() {
    document.getElementById("popup").style.display = "block";
  }
  
  $(document).ready(function() {
    $.ajax({
      type: "GET",  
      url: "https://raw.githubusercontent.com/ming-zhao/ming-zhao.github.io/master/data/course.csv",
      dataType: "text",       
      success: function(data)  
      {
        var items = data.split(',');
        var url = window.location.pathname;
        var filename = url.substring(url.lastIndexOf('/')+1);
        for (var i = 0, len = items.length; i < len; ++i) {
            if (items[i].includes(filename)) {
                password=items[i+1];
                break;
            }
        }
        var code_blocks = document.getElementsByClassName('nbinput docutils container');
        if (code_blocks[0]==null) { 
            isHtml=0;
            code_blocks = document.getElementsByClassName('input');
            class_output=".output_wrapper";
            class_input="input";
            class_answer='.cell';
        }
        else { 
            isHtml=1;
            class_output=".nboutput";
            class_input="nbinput";
            class_answer=".nboutput";
        }
        
        for (var i = 0, len = code_blocks.length; i < len; ++i) {
          if (
              code_blocks[i].innerHTML.indexOf("toggle") !== -1 
              || code_blocks[i].innerHTML.indexOf("button onclick") !== -1
             ) {
            code_blocks[i].style.display = "none";
          }
        }
        for (var i = 0, len = code_blocks.length; i < len; ++i) {
          if (code_blocks[i].innerHTML.indexOf("hide_answer") !== -1) {
            code_blocks[i].style.display = "none";
            if (isHtml) {
              code_blocks[i].nextElementSibling.nextElementSibling.style.display = "none";
            }
            else{
              code_blocks[i].closest(class_answer).nextElementSibling.style.display = "none";
            }            
          }
        }
      }
    });
  });
</script>''', raw=True)


def hide_answer():
    html = """
        <a href="#" onclick="hide_answer(this); return false;">show answer</a>
    """
    return HTML(html)   

def toggle():
    html = """
        <a href="#" onclick="code_toggle(this); return false;">show code</a>
    """
    return HTML(html)