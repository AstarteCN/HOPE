# Stage 3 TensorBoard Summary

- Log dir: `D:\Github\HOPE\src\log\exp\sac_ogm_stage4_ogm_proxy_1k_20260620_234404`
- Event dir: `D:\Github\HOPE\src\log\exp\sac_ogm_stage4_ogm_proxy_1k_20260620_234404`
- Episode count: `1000`
- Required minimum episodes: `1000`
- Training budget met: `True`
- Environment step count: `131691`
- Estimated SAC updates after warmup: `12145`
- Has nonfinite watched scalar: `True`
- Hard-reject has nonfinite scalar: `False`
- Nonfinite scalar tags: `['success_rate_Complex', 'success_rate_Extrem', 'success_rate_dlp']`
- Hard-reject nonfinite scalar tags: `[]`

## Watched Scalars

- `total_reward`: count `1000`, first step `0`, last step `999`, first `5.862369537353516`, last `5.5178728103637695`, mean `2.2188345155345885`, min `-6.319822311401367`, max `6.347952365875244`, has nonfinite `False`, mean first500 `1.953887856822461`, mean last100 `2.6605539082968606`, mean last500 `2.4837811742467166`, trend delta `1.3066282649803904`
- `avg_reward`: count `1000`, first step `0`, last step `999`, first `0.05804326385259628`, last `0.026392538100481033`, mean `0.01758609444857575`, min `-0.019196191802620888`, max `0.06368353962898254`, has nonfinite `False`, mean first500 `0.014866564170457423`, mean last100 `0.02435562494036276`, mean last500 `0.020305624726694078`, trend delta `0.006285681064473464`
- `actor_loss`: count `607`, first step `86`, last step `998`, first `0.02664119005203247`, last `-0.49679118394851685`, mean `-0.24157193823083867`, min `-0.7517157793045044`, max `0.0867350846529007`, has nonfinite `False`, mean first500 `-0.2112670042756945`, mean last100 `-0.3883056816458702`, mean last500 `-0.2603073806334287`, trend delta `-0.2827609353512526`
- `critic_loss`: count `607`, first step `86`, last step `998`, first `0.047548793256282806`, last `0.04703424125909805`, mean `0.05366453920265497`, min `0.0061156125739216805`, max `0.9920406341552734`, has nonfinite `False`, mean first500 `0.053189892215654254`, mean last100 `0.05711714868433773`, mean last500 `0.0480850071106106`, trend delta `-0.07103633355349302`
- `action_std0`: count `1000`, first step `0`, last step `999`, first `-0.0`, last `0.021287735551595688`, mean `0.01270889059522915`, min `-2.434200723655522e-05`, max `0.021766595542430878`, has nonfinite `False`, mean first500 `0.005992033846187951`, mean last100 `0.02134551279246807`, mean last500 `0.01942574734427035`, trend delta `0.02129246186465025`
- `action_std1`: count `1000`, first step `0`, last step `999`, first `-0.0`, last `-0.018952498212456703`, mean `-0.007812583492907834`, min `-0.018952498212456703`, max `2.0429521100595593e-05`, has nonfinite `False`, mean first500 `-0.002804692984236681`, mean last100 `-0.017737829443067312`, mean last500 `-0.012820474001578987`, trend delta `-0.01831939309835434`
- `alpha`: count `1000`, first step `0`, last step `999`, first `0.009999999776482582`, last `0.009413504041731358`, mean `0.009725913702510297`, min `0.009413504041731358`, max `0.009999999776482582`, has nonfinite `False`, mean first500 `0.009883885001763702`, mean last100 `0.009444308346137404`, mean last500 `0.009567942403256893`, trend delta `-0.0005710386112332336`
- `success_rate_Normal`: count `1000`, first step `0`, last step `999`, first `1.0`, last `0.7300000190734863`, mean `0.732266104221344`, min `0.6666666865348816`, max `1.0`, has nonfinite `False`, mean first500 `0.7351722096204758`, mean last100 `0.7150000071525574`, mean last500 `0.7293599988222123`, trend delta `-0.1633991134166718`
- `success_rate_Complex`: count `1000`, first step `0`, last step `999`, first `None`, last `0.6499999761581421`, mean `0.5676726129081275`, min `0.0`, max `0.75`, has nonfinite `True`, mean first500 `0.5377898806333542`, mean last100 `0.6403999876976013`, mean last500 `0.597579999923706`, trend delta `0.09360956907272333`
- `success_rate_Extrem`: count `1000`, first step `0`, last step `999`, first `None`, last `0.12999999523162842`, mean `0.08772604550815417`, min `0.0`, max `0.1599999964237213`, has nonfinite `True`, mean first500 `0.0594611861333251`, mean last100 `0.13839999973773956`, mean last500 `0.11592000070214271`, trend delta `0.13539999812841416`
- `success_rate_dlp`: count `1000`, first step `0`, last step `999`, first `None`, last `0.46000000834465027`, mean `0.37008795952246926`, min `0.0`, max `0.47999998927116394`, has nonfinite `True`, mean first500 `0.303235396027565`, mean last100 `0.4670999950170517`, mean last500 `0.43703999519348147`, trend delta `0.3302276772260666`
- `step_num`: count `1000`, first step `0`, last step `999`, first `101.0`, last `10.0`, mean `131.691`, min `6.0`, max `200.0`, has nonfinite `False`, mean first500 `131.41`, mean last100 `123.47`, mean last500 `131.972`, trend delta `19.779999999999987`

## Warnings

- success_rate_Extrem remains below 0.2 at the end of the run
