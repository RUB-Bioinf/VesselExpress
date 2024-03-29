$(document).ready(function () {
    // vesselness fields
    var isCheckedVes1 = $("#vesselness_1").prop('checked');
    var isCheckedVes2 = $("#vesselness_2").prop('checked');
    if (isCheckedVes2 === true){
        document.getElementById("gamma_2").disabled = false;
            document.getElementById("gamma_2").style.color = 'black';
            document.getElementById("sigma_2").disabled = false;
            document.getElementById("sigma_2").style.color = 'black';
            document.getElementById("cutoff_method_2").disabled = false;
            document.getElementById("cutoff_method_2").style.color = 'black';
        }
    else {
        document.getElementById("gamma_2").disabled = true;
        document.getElementById("gamma_2").style.color = '#ababab';
        document.getElementById("sigma_2").disabled = true;
        document.getElementById("sigma_2").style.color = '#ababab';
        document.getElementById("cutoff_method_2").disabled = true;
        document.getElementById("cutoff_method_2").style.color = '#ababab';
    }
    if (isCheckedVes1 === true){
        document.getElementById("gamma_1").disabled = false;
            document.getElementById("gamma_1").style.color = 'black';
            document.getElementById("sigma_1").disabled = false;
            document.getElementById("sigma_1").style.color = 'black';
            document.getElementById("cutoff_method_1").disabled = false;
            document.getElementById("cutoff_method_1").style.color = 'black';
        }
    else {
        document.getElementById("gamma_1").disabled = true;
        document.getElementById("gamma_1").style.color = '#ababab';
        document.getElementById("sigma_1").disabled = true;
        document.getElementById("sigma_1").style.color = '#ababab';
        document.getElementById("cutoff_method_1").disabled = true;
        document.getElementById("cutoff_method_1").style.color = '#ababab';
    }

    // Executed on page load. Define standard behaviour of objects in here.
    $('.message .close').on('click', function () {
        $(this).closest('.message').transition('fade');
    });

    $("#pixelsizetooltip").click(function (event) {
            event.preventDefault();
            $('#pixelinfo').modal('show');
    });

    // Inputs
    $('input[type="checkbox"]').on('change', function () {
        this.value = this.checked ? 1 : 0;
    }).change();

    $("form").submit(function () {
        $(this).find('input[type="checkbox"]').each(function () {
            var checkbox_this = $(this);
            if (checkbox_this.is(":checked") == true) {
                checkbox_this.attr('value', '1');
            } else {
                checkbox_this.prop('checked', true);
                checkbox_this.attr('value', '0');
            }
        })
    });

    // Pixel label
    $("#pixel_dims").text('1 pixel/voxel = ' +
        document.getElementById('dim_x').value + ' x ' +
        document.getElementById('dim_y').value + ' x ' +
        document.getElementById('dim_z').value + ' µm');


})

// flash message handler
function flash_handler(flash_collection) {
    /* flash_collection is a list with n dictionaries that each  have 'message' and 'type' keys.
    Each dictionary in the list produces one flash message.
    */
    for (var i = 0; i < flash_collection.length; i++) {
        if (flash_collection[i].type == 'success') {
            $('body').toast({
                showIcon: 'check circle outline',
                message: flash_collection[i].message,
                showProgress: 'top',
                classProgress: 'green',
                progressUp: true,
                pauseOnHover: true,
                displayTime: 3000
            })
        } else if (flash_collection[i].type == 'error') {
            $('body').toast({
                showIcon: 'times circle outline',
                message: flash_collection[i].message,
                showProgress: 'top',
                classProgress: 'red',
                progressUp: true,
                pauseOnHover: true,
                displayTime: 3000
            })
        } else if (flash_collection[i].type == 'warning') {
            $('body').toast({
                showIcon: 'question circle outline',
                message: flash_collection[i].message,
                showProgress: 'top',
                classProgress: 'yellow',
                progressUp: true,
                pauseOnHover: true,
                displayTime: 3000
            })
        } else if (flash_collection[i].type == 'finish') {
            $('body').toast({
                showIcon: 'check circle outline',
                message: flash_collection[i].message,
                showProgress: 'top',
                classProgress: 'green',
                displayTime: 0
            })
        }
    }
}



