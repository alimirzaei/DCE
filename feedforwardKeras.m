
function a = feedforwardKeras(image)
    thingSpeakURL = 'http://localhost:5000/estimate_channel';
    addpath('matlab-json')
    json.startup
    data =json.dump(image);
    options = weboptions('MediaType','application/json');
    response = webwrite(thingSpeakURL,data,options);
    a = json.load(response);
end