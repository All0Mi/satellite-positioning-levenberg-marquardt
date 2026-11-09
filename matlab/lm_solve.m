function [p, resnorm, residual, exitflag, output] = lm_solve(resFun, jacFun, p0, opts)

    if nargin < 4, opts = struct(); end
    defOpts = struct('FunctionTolerance', 1e-10, 'StepTolerance', 1e-10, ...
        'MaxIterations', 500, 'Display', 'off', 'lambda0', 1e-3);
    fn = fieldnames(defOpts);
    for i = 1:numel(fn)
        if ~isfield(opts, fn{i}) || isempty(opts.(fn{i}))
            opts.(fn{i}) = defOpts.(fn{i});
        end
    end

    p = p0(:);
    r = resFun(p);
    J = jacFun(p);
    f = 0.5*(r'*r);
    lambda = opts.lambda0;
    nu = 2;
    funcCount = 1;
    exitflag = 0;
    history = zeros(0, 4);   % [iter, f(p), ||step||, lambda]
    history(1, :) = [0, f, 0, lambda];

    if strcmpi(opts.Display, 'iter')
        fprintf('%5s %16s %16s %10s\n', 'iter', 'f(p)', '||step||', 'lambda');
    end

    for it = 1:opts.MaxIterations
        H = J'*J;
        g = J'*r;
        diagH = diag(H);
        diagH(diagH == 0) = 1;

        step_ok = false;
        while ~step_ok
            A = H + lambda*diag(diagH);
            delta = -A \ g;
            p_new = p + delta;
            r_new = resFun(p_new);
            f_new = 0.5*(r_new'*r_new);
            funcCount = funcCount + 1;

            if f_new < f
                step_ok = true;
            else
                lambda = lambda * nu;
                if lambda > 1e12
                    % damping saturated without improving f -> give up on this step
                    step_ok = true;
                    p_new = p; r_new = r; f_new = f; delta = zeros(size(p));
                end
            end
        end

        stepNorm = norm(delta);
        relFchange = abs(f_new - f) / max(1, abs(f));
        history(end+1, :) = [it, f_new, stepNorm, lambda];

        if strcmpi(opts.Display, 'iter')
            fprintf('%5d %16.6e %16.6e %10.2e\n', it, f_new, stepNorm, lambda);
        end

        if f_new < f
            lambda = lambda / nu;
        end

        p_prev = p;
        p = p_new; r = r_new; f = f_new; J = jacFun(p);

        if stepNorm < opts.StepTolerance * (norm(p_prev) + opts.StepTolerance)
            exitflag = 1; break;
        end
        if relFchange < opts.FunctionTolerance
            exitflag = 2; break;
        end
    end

    residual = r;
    resnorm = r'*r;
    output.iterations = it;
    output.funcCount  = funcCount;
    output.history    = history;
end
