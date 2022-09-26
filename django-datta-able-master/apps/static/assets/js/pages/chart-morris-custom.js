'use strict';
$(document).ready(function() {
    setTimeout(function() {
    // [ bar-simple ] chart start
    Morris.Bar({
        element: 'morris-bar-chart',
        data: [{
                y: 'Less Than 24',
                a: 202,
                b: 89,
                c: 81,
            },
            {
                y: '24 - 44',
                a: 3353,
                b: 1397,
                c: 1721,
            },
            {
                y: '45 - 64',
                a: 1858,
                b: 1107,
                c: 1404,
            },
            {
                y: '65+',
                a: 393,
                b: 19,
                c: 35,
            },
            {
                y: 'unspecified',
                a: 624,
                b: 179,
                c: 340,
            }
        ],
        xkey: 'y',
        barSizeRatio: 0.70,
        barGap: 3,
        resize: true,
        responsive:true,
        ykeys: ['a', 'b', 'c'],
        labels: ['Total', 'Africans', 'First Time'],
        barColors: ["0-#1de9b6-#1dc4e9", "0-#899FD4-#A389D4", "#04a9f5"]
    });
    // [ bar-simple ] chart end

    // [ bar-stacked ] chart start
    Morris.Bar({
        element: 'morris-bar-stacked-chart',
        data: [{
                y: 'Business',
                a: 397,
                b: 255,
                c: 652,
            },
            {
                y: 'Holidays',
                a: 559,
                b: 466,
                c: 1025,
            },
            {
                y: 'Meeting',
                a: 310,
                b: 448,
                c: 758,
            },
            {
                y: 'Visit Family',
                a: 561,
                b: 503,
                c: 1064,
            },
            {
                y: 'Other',
                a: 516,
                b: 398,
                c: 914,
            }
        ],
        xkey: 'y',
        stacked: true,
        barSizeRatio: 0.50,
        barGap: 3,
        resize: true,
        responsive:true,
        ykeys: ['a', 'b', 'c'],
        labels: ['Male', 'Female', 'Total'],
        barColors: ["0-#1de9b6-#1dc4e9", "0-#899FD4-#A389D4", "#04a9f5"]
    });
    // [ bar-stacked ] chart end

    // [ area-angle-chart ] start
    Morris.Area({
        element: 'morris-area-chart',
        data: [{
                y: '1940',
                a: 227,
                b: 251
            },
            {
                y: '1958',
                a: 427,
                b: 308
            },
            {
                y: '1978',
                a: 895,
                b: 760
            },
            {
                y: '1998',
                a: 2630,
                b: 2909
            },
            {
                y: '2022',
                a: 819,
                b: 1098
            },
            
        ],
        xkey: 'y',
        ykeys: ['a', 'b'],
        labels: ['Female', 'Male'],
        pointSize: 0,
        fillOpacity: 0.6,
        pointStrokeColors: ['black', '#A389D4'],
        behaveLikeLine: true,
        gridLineColor: '#e0e0e0',
        lineWidth: 0,
        smooth: false,
        hideHover: 'auto',
        responsive:true,
        lineColors: ['#FF78FF', '#A389D4'],
        resize: true
    });
    // [ area-angle-chart ] end

    // [ area-smooth-chart ] start
    Morris.Area({
        element: 'morris-area-curved-chart',
        data: [{
            period: '0 nights',
            iphone: 2902,
            ipad: 1043,
            itouch: 780
        }, {
            period: '1 night',
            iphone: 978,
            ipad: 659,
            itouch: 1657
        }, {
            period: '2 nights',
            iphone: 223,
            ipad: 152,
            itouch: 659
        }, {
            period: '3 nights',
            iphone: 67,
            ipad: 12,
            itouch: 1039
        }, {
            period: '4 nights',
            iphone: 32,
            ipad: 2,
            itouch: 129
        }, {
            period: '5 and more',
            iphone: 85,
            ipad: 21,
            itouch: 146
        }],
        lineColors: ['#A389D4', '#1de9b6', '#04a9f5'],
        xkey: 'period',
        ykeys: ['iphone', 'ipad', 'itouch'],
        labels: ['Night in Arbaminch', 'Night in Gamo Gofa', 'Total Nights'],
        pointSize: 0,
        lineWidth: 0,
        resize: true,
        fillOpacity: 0.8,
        responsive:true,
        behaveLikeLine: true,
        gridLineColor: '#d2d2d2',
        hideHover: 'auto'
    });
    // [ area-smooth-chart ] end

    // [ line-angle-chart ] Start
    Morris.Line({
        element: 'morris-line-chart',
        data: [{
                y: '1940',
                a: 4890,
                b: 5828
            },
            {
                y: '1958',
                a: 13872,
                b: 16721
            },
            {
                y: '1978',
                a: 26173,
                b: 24736
            },
            {
                y: '1998',
                a: 12589,
                b: 13848
            },
            {
                y: '2022',
                a: 9589,
                b: 9848
            },
            
        ],
        xkey: 'y',
        redraw: true,
        resize: true,
        smooth: false,
        ykeys: ['a', 'b'],
        hideHover: 'auto',
        responsive:true,
        labels: ['Female', 'Male'],
        lineColors: ['#1de9b6', '#04a9f5']
    });
    // [ line-angle-chart ] end
    // [ line-smooth-chart ] start
    Morris.Line({
        element: 'morris-line-smooth-chart',
        data: [{
            y: '1940',
            a: 4890,
            b: 5828
        },
        {
            y: '1958',
            a: 13872,
            b: 16721
        },
        {
            y: '1978',
            a: 26173,
            b: 24736
        },
        {
            y: '1998',
            a: 12589,
            b: 13848
        },
        {
            y: '2022',
            a: 9589,
            b: 9848
        },

        ],
        xkey: 'y',
        redraw: true,
        resize: true,
        ykeys: ['a', 'b'],
        hideHover: 'auto',
        responsive:true,
        labels: ['Female', 'Male'],
        lineColors: ['#1de9b6', '#A389D4']
    });
    // [ line-smooth-chart ] end

    // [ Donut-chart ] Start
    var graph = Morris.Donut({
        element: 'morris-donut-chart',
        data: [{
                value: 1605,
                label: 'Excellent Experience'
            },
            {
                value: 520,
                label: 'Frindly People'
            },
            {
                value: 428,
                label: 'Good Service'
            },
            {
                value: 713,
                label: 'No comment'
            },
            {
                value: 517,
                label: 'others'
            }
        ],
        colors: [
            '#1de9b6',
            '#A389D4',
            '#04a9f5',
            '#1dc4e9',
            '#F367e3'
        ],
        resize: true,
        formatter: function(x) {
            return "val : " + x
        }
    });    
    // [ Donut-chart ] end
        }, 700);
});
