function z = isaextractdata(u)
    if isa(u, 'dlarray')
        z = extractdata(u);
    else
        z = u;
    end
end