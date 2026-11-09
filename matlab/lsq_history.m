function stop = lsq_history(~, optimValues, state)
    persistent hist
    stop = false;

    switch state
        case 'init'
            hist = zeros(0, 2);
        case 'iter'
            hist(end+1, :) = [optimValues.iteration, optimValues.resnorm];
        case 'done'
            assignin('base', 'lsqHistory', hist);
    end
end
