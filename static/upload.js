var blah = document.getElementById('blah');

function readURL(input) { 
if (input.files && input.files[0]) { 
    var reader = new FileReader(); 

    reader.onload = function (e) { 
        $('#blah,#bleh') 
            .attr('src', e.target.result) 
            .width(150) 
            .height(200); 
    }; 
    blah.src = input.files
    reader.readAsDataURL(input.files); 
} 
} 

