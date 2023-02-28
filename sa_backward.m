function z = sa_backward(E)
    a0 = 20;
    g  = E;

    if (g < 0.0001)
        z = single(0.0001);
    else
        q = abs(sa_forward(E))/g;
        z = single(q * exp(1+(a0*g*g)/(1+g^2)^2));
    end
end