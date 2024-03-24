function accuracy = test_Chow_Liu(test_data, test_labels, nclass, prior, P_X, P_XY, mst)

    [m, n] = size(test_data); % n is number of features

    correct = 0;
    for test_i = 1:m
        test_sample = test_data(test_i, :);
        target_class = test_labels(test_i);

        % Need to compute posterior probabilities for each class
        posterior = zeros(nclass, 1);

        for a_class = 1:nclass
            
            % We will caculate in the log domain; otherwise there is a risk of underflow
            posterior(a_class) = log(prior(a_class));

            % For each edge in the class conditoinal Tree T1, add the joint probability
            T = squeeze(mst(a_class, :, :));
            for i = 1:size(T, 1)
                edge = T(i, :);
                % we need the first vertex to be smaller than the second 
                if edge(1) < edge(2)
                    X = edge(1);
                    Y = edge(2);
                else
                    X = edge(2);
                    Y = edge(1);
                end

                % Find which values the features X & Y take in the test sample
                valx = test_sample(X); 
                valy = test_sample(Y); 

                % Find corresponding joint probability 
                posterior(a_class) = posterior(a_class) + log(P_XY(a_class, X, Y, valx, valy));

            end

            % Compute the frequency of each vertex in the Tree
            freq = zeros(n, 1);  % initialize frequency to 0
            t = T(:);
            for i = 1:numel(t)
                vertex = t(i);
                freq(vertex) = freq(vertex) + 1;
            end


            % Next subtract the individual log of probabilities P(X) (freq-1) times
            for X = 1:n
                f = freq(X);
                if f > 1
                    % Find the value feature X takes in the test sample
                    valx = test_sample(X); % either 1 or 2 or 3 etc.
                    posterior(a_class) = posterior(a_class) - (f-1) * log(P_X(a_class, X, valx));
                end
            end

        end

        % Find the class with the largest posterior (argmax)
        [maxval, predicted_class] = max(posterior);

        if predicted_class == target_class
            correct = correct + 1;
        end
    
    end
    accuracy = correct * 100 / m;
end