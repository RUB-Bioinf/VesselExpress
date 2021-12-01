$(document).ready(function () {
    $('.selection.dropdown').dropdown();

    $('.ui.dropdown').dropdown();

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

    // Flashes
    $('#successtoast')
        .toast({
            title: 'Success',
            class: 'green',
            showProgress: 'top',
            progressUp: true,
            pauseOnHover: true,
            displayTime: 3000
        });
    $('#errortoast')
        .toast({
            title: 'Error',
            class: 'red',
            showProgress: 'top',
            progressUp: true,
            pauseOnHover: true,
            displayTime: 3000
        });
    $('#warningtoast')
        .toast({
            title: 'Warning',
            class: 'yellow',
            showProgress: 'top',
            progressUp: true,
            pauseOnHover: true,
            displayTime: 3000
        });
})



