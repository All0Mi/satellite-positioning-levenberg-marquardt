function [F, J] = gps_model(p, Sat, d)
    F = gps_residuals(p, Sat, d);
    if nargout > 1
        J = gps_jacobian(p, Sat);
    end
end
