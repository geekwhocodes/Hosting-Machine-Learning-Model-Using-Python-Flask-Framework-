$(document).ready(function(){

    $("#submit_text").on('click', (function (e) {    
        $.ajax({
            url: "/v1/profanity/prediction",
            type: "POST",             
            data: JSON.stringify({'text':$("#input").val()}),
            contentType: 'application/json',    
            success: function (data) 
            {
                $('#loading').hide();
                $("#message").text("Result : " + data.result);
                
            },error: function(XMLHttpRequest, textStatus, errorThrown) { 
                $("#message").text(errorThrown);
            }
        }); 
    
    }));
});