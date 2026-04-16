$tests = @(
    @{in="i have an idea";                                                       expect="clarifying"},
    @{in="hello";                                                                expect="chat"},
    @{in="hey";                                                                  expect="chat"},
    @{in="give me feedback";                                                     expect="clarifying"},
    @{in="i want to build something";                                            expect="clarifying"},
    @{in="i am thinking about a startup";                                        expect="clarifying"},
    @{in="what do you think";                                                    expect="clarifying"},
    @{in="A telemedicine platform for patients in Dubai to book doctors online"; expect="analysis"},
    @{in="An invoice financing app for Egyptian SMEs to get paid faster";        expect="analysis"},
    @{in="SaaS CRM for small businesses in Saudi Arabia to manage sales";        expect="analysis"}
)
$pass=0; $fail=0
foreach ($t in $tests) {
    $body = '{"context":{},"messages":[{"role":"user","content":"' + $t.in + '"}]}'
    try {
        $r = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/interact" -ContentType "application/json" -Body $body -TimeoutSec 30
        $got = $r.type
    } catch { $got = "TIMEOUT" }
    $ok = $got -eq $t.expect
    if ($ok) { $pass++ } else { $fail++ }
    $icon = if ($ok) { "PASS" } else { "FAIL" }
    Write-Host "[$icon] exp=$($t.expect.PadRight(10)) got=$($got.PadRight(10)) | $($t.in)"
}
Write-Host ""
Write-Host "=== $pass/$($tests.Count) PASSED ==="
