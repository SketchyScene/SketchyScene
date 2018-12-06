function showEmail() {
	$("#vemail").html("aaronzou1125" + String.fromCharCode(64) + "gmail" + String.fromCharCode(46) + "com<br/>" + 
	"me" + String.fromCharCode(64) + "duruofei" + String.fromCharCode(46) + "com<br/>" +
	"mohaoran1995" + String.fromCharCode(64) + "gmail" + String.fromCharCode(46) + "com<br/>");
}

$( document ).ready(function() {
    $('a[href^="http://"], a[href^="https://"]').attr('target','_blank');

	$("#vemail").on({ 'touchstart' : function() {
		showEmail();
	} });
	$("#vemail").click(function() {
		showEmail();
	});
	
	if ($("#actived").val() == "people") {
		$("#mhome").removeClass("active");
		$("#mpeople").addClass("active");
	}
});
