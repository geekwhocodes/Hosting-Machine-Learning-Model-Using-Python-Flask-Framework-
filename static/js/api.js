(function(){
    'use strict';
    var restService = {}

    
    restService.getStats = function(){
        $.ajax({url: "../../v1/stats", success: function(result){
            return results;
        }});
    }
    


})();