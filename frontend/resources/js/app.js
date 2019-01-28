angular.module('shanyNet', [])
.controller('ShanyNetController', ['$rootScope', '$scope','fileUpload', function($rootScope, $scope, fileUpload) {
    this.isRunning = false;
    this.results = [];

    $rootScope.$on('finishedInference', function (event, data) {
        this.isRunning = false;
        this.results = data;
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
}])
.directive('fileModel', ['$parse', function ($parse) {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            var model = $parse(attrs.fileModel);
            var modelSetter = model.assign;

            element.bind('change', function() {
                scope.$apply(function() {
                    modelSetter(scope, element[0].files[0]);
                });
            });
        }
    };
}])
.service('fileUpload', ['$rootScope','$http', function ($rootScope, $http) {
    this.uploadFileToUrl = function(file, uploadUrl) {
        var fd = new FormData();
        fd.append('file', file);

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