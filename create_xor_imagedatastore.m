function z = create_xor_imagedatastore(labelfile, location)
    imds = imageDatastore(location);
    labels = load(labelfile).lbls;
    imds.Labels = labels.';
    z = imds;
end