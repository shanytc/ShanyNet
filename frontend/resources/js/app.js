angular.module('shanyNet', [])
.controller('ShanyNetController', ['$rootScope', '$scope','fileUpload', function($rootScope, $scope, fileUpload) {
    this.isRunning = false;
    this.readyToSearch = false;
    this.results = [];

    $rootScope.$on('finishedInference', function (event, data) {
        this.isRunning = false;
        this.results = data;
    }.bind(this));

    $rootScope.$on('readyToUpload', function () {
        this.readyToSearch = true;
    }.bind(this));

    this.uploadFile = function() {
        var file = $scope.myFile;
        this.isRunning = true;
        this.results = [];

        //console.dir(file);
        var uploadUrl = "http://iltlvl914:5005/upload";
        fileUpload.uploadFileToUrl(file, uploadUrl);
    };

    this.getResults = function(){
        return this.results;
    };

    this.getIsRunning = function(){
        return this.isRunning;
    };

    this.getImageLink = function(image){
        return 'http://iltlvl914:5005/image/' + image[3] + '/' + encodeURI(image[2])
    }
}])
.directive('fileModel', ['$rootScope','$parse', function ($rootScope, $parse) {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            var model = $parse(attrs.fileModel);
            var modelSetter = model.assign;
            var fr=new FileReader();

            element.bind('change', function() {
                scope.$apply(function() {
                    $rootScope.$emit('readyToUpload');
                    fr.onload = function(e) {
                        $('#seletedImage')[0].src = this.result;
                    };
                    fr.readAsDataURL(element[0].files[0]);
                    modelSetter(scope, element[0].files[0]);
                });
            });
        }
    };
}])
.service('fileUpload', ['$rootScope','$http', function ($rootScope, $http) {
    this.uploadFileToUrl = function(file, uploadUrl) {
        var algo = $('input[type=radio][name=algo]:checked').val();
        var search = $('input[type=radio][name=search]:checked').val();
        var results = $('#num_of_results').val();

        var fd = new FormData();
        fd.append('file', file);
        fd.append('algo', algo);
        fd.append('search', search);
        fd.append('results', results);

        $http.post(uploadUrl, fd, {
            transformRequest: angular.identity,
            headers: {'Content-Type': undefined}
        }).then(function(response){
            var inference_data = response.data.data;
            $rootScope.$emit('finishedInference', inference_data);
        },function(error){
            $rootScope.$emit('finishedInference', []);
        });
    }
}]);