import os
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
QUESTIONS_FILE = BASE_DIR / "data" / "questions.csv"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 50 Detailed Space Exploration & Astrobiology Documents
DOCS = {
    "apollo_11_mission": """# Apollo 11 Mission Overview
Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969, at 20:17 UTC. Neil Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later, on July 21 at 02:56 UTC. Aldrin joined him 19 minutes later. They spent about two hours and 15 minutes together outside the spacecraft in Tranquility Base.
The Command Module Columbia was piloted by Michael Collins in lunar orbit. Armstrong and Aldrin collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. The mission was launched by a Saturn V rocket from Kennedy Space Center on Merritt Island, Florida, on July 16 at 13:32 UTC. The spacecraft returned to Earth and splashed down in the Pacific Ocean on July 24, 1969, after 8 days, 3 hours, 18 minutes, and 35 seconds of flight.""",

    "voyager_1_probe": """# Voyager 1 Deep Space Mission
Voyager 1 is a space probe launched by NASA on September 5, 1977, as part of the Voyager program to study the outer Solar System and interstellar space. Launched 16 days after its twin, Voyager 2, Voyager 1 communicated through NASA's Deep Space Network to transmit scientific data.
At a distance of 162.7 AU (24.3 billion km) from Earth as of 2024, it is the most distant human-made object from Earth. The probe made flybys of Jupiter in March 1979 and Saturn in November 1980, discovering active volcanoes on Jupiter's moon Io and intricate structures in Saturn's rings.
Voyager 1 crossed the heliopause and entered interstellar space on August 25, 2012. It carries the Golden Record, a 12-inch gold-plated copper disk containing sounds and images selected to portray the diversity of life and culture on Earth. Its primary electrical power is supplied by three radioisotope thermoelectric generators (RTGs) fueled by plutonium-238.""",

    "voyager_2_probe": """# Voyager 2 Grand Tour Mission
Voyager 2 is a space probe launched by NASA on August 20, 1977, to study the outer planets. It was launched before Voyager 1 on a trajectory that took longer to reach Jupiter and Saturn but enabled further encounters with Uranus and Neptune. Voyager 2 remains the only spacecraft to have visited either of the ice giant planets: Uranus (closest approach January 24, 1986) and Neptune (closest approach August 25, 1989).
During its Uranus encounter, Voyager 2 discovered 10 new moons and two new rings. At Neptune, it discovered five moons, four rings, and Neptune's 'Great Dark Spot'. Voyager 2 crossed the heliopause into interstellar space on November 5, 2018, at a distance of 119.7 AU. Like its twin, it carries three radioisotope thermoelectric generators and the Voyager Golden Record.""",

    "curiosity_rover_mars": """# Mars Science Laboratory (Curiosity Rover)
The Curiosity rover, part of NASA's Mars Science Laboratory (MSL) mission, landed in Gale Crater on Mars on August 6, 2012. The rover's primary mission was to investigate the Martian climate and geology, and to assess whether the Gale Crater area ever had an environment capable of supporting microbial life.
Curiosity weighs 899 kg (1,982 lb) and is powered by a Multi-Mission Radioisotope Thermoelectric Generator (MMRTG) using plutonium-238 dioxide. Key instruments include the ChemCam laser spectrometer, the Sample Analysis at Mars (SAM) instrument suite, the Mastcam stereo cameras, and the Alpha Particle X-Ray Spectrometer (APXS). In 2013, Curiosity discovered evidence of an ancient freshwater lake inside Gale Crater that contained all the basic chemical ingredients essential for life (carbon, hydrogen, oxygen, nitrogen, phosphorus, and sulfur).""",

    "perseverance_rover_mars": """# Mars 2020 Perseverance Rover
Perseverance is a car-sized Mars rover designed to explore Jezero crater on Mars as part of NASA's Mars 2020 mission. It was manufactured by the Jet Propulsion Laboratory and launched on July 30, 2020. Perseverance landed successfully on Mars on February 18, 2021.
The rover carries seven primary scientific instruments, including SuperCam, PIXL, SHERLOC, and the MOXIE experiment, which demonstrated the production of breathable oxygen from Martian carbon dioxide on April 20, 2021.
Perseverance also carried the Ingenuity helicopter, a technology demonstration drone that completed 72 flights between April 2021 and January 2024 before sustaining rotor damage. Perseverance is collecting rock core samples to be returned to Earth in a future Mars Sample Return mission.""",

    "james_webb_space_telescope": """# James Webb Space Telescope (JWST)
The James Webb Space Telescope is a space telescope designed primarily to conduct infrared astronomy. High-resolution and high-sensitivity instruments allow it to view objects too old, distant, or faint for the Hubble Space Telescope.
JWST was launched on December 25, 2021, on an Ariane 5 rocket from Kourou, French Guiana, and arrived at the Sun-Earth L2 Lagrange point in January 2022, approximately 1.5 million kilometers (930,000 miles) from Earth.
The telescope features a 6.5-meter-diameter primary mirror consisting of 18 hexagonal beryllium segments coated with gold. Its primary instruments include the Near-Infrared Camera (NIRCam), Near-Infrared Spectrograph (NIRSpec), Mid-Infrared Instrument (MIRI), and the Fine Guidance Sensor/Near InfraRed Imager and Slitless Spectrograph (FGS/NIRISS). JWST operates at cryogenic temperatures below 50 Kelvin (-223 degrees Celsius).""",

    "hubble_space_telescope": """# Hubble Space Telescope (HST)
The Hubble Space Telescope was launched into low Earth orbit on April 24, 1990, aboard Space Shuttle Discovery (STS-31) and remains in operation. Hubble's orbit outside the distortion of Earth's atmosphere allows it to capture extremely high-resolution images with substantially lower background light than ground-based telescopes.
Hubble has a 2.4-meter (7.9 ft) primary mirror, and its four main instruments observe in the ultraviolet, visible, and near-infrared regions of the electromagnetic spectrum. Shortly after launch, scientists discovered a spherical aberration in the primary mirror due to flawed manufacturing. In December 1993, the STS-61 servicing mission installed corrective optics (COSTAR) and the WFPC2 camera, restoring the telescope to its intended optical capability. Hubble has completed over 1.5 million observations.""",

    "chandrayaan_1_mission": """# Chandrayaan-1 Lunar Mission
Chandrayaan-1 was India's first lunar probe, launched by the Indian Space Research Organisation (ISRO) on October 22, 2008, using a Polar Satellite Launch Vehicle (PSLV-C11) from the Satish Dhawan Space Centre at Sriharikota.
The mission operated until August 2009. The spacecraft carried 11 scientific instruments: five Indian and six from foreign agencies including NASA and ESA.
On November 14, 2008, the Moon Impact Probe (MIP) separated from the orbiter and struck the lunar south pole near Shackleton Crater. Data from the Moon Mineralogy Mapper (M3), a NASA-supplied imaging spectrometer on Chandrayaan-1, led to the historic confirmation of widespread water ice and hydroxyl molecules on the lunar surface.""",

    "chandrayaan_3_mission": """# Chandrayaan-3 Historic Lunar Landing
Chandrayaan-3 was the third mission in the Chandrayaan programme developed by ISRO. Launched on July 14, 2023, aboard an LVM3-M4 rocket from Satish Dhawan Space Centre, the mission aimed to demonstrate safe soft landing and roving on the lunar surface.
On August 23, 2023, at 18:04 IST (12:34 UTC), the Vikram lander successfully touched down near the lunar south polar region at 69.37 degrees South latitude and 32.35 degrees East longitude (named Shiv Shakti Point). This made India the first nation to land successfully in the lunar south polar area and the fourth nation overall to achieve a soft lunar landing.
The Pragyan rover was deployed shortly after landing and operated for one lunar day (approximately 14 Earth days), detecting sulfur, aluminum, calcium, iron, and titanium on the lunar surface using LIBS and APXS spectrometers.""",

    "mangalyaan_mars_orbiter": """# Mars Orbiter Mission (Mangalyaan)
The Mars Orbiter Mission (MOM), unofficially known as Mangalyaan, was India's first interplanetary mission. Launched on November 5, 2013, by ISRO using a PSLV-XL rocket (C25), the probe successfully entered Mars orbit on September 24, 2014.
This made ISRO the fourth space agency to reach Mars orbit (after Roscosmos, NASA, and ESA) and the first space agency in the world to reach Mars on its maiden attempt.
The spacecraft cost approximately $74 million USD (Rs 450 crore), making it one of the most cost-effective interplanetary missions ever executed. It carried five scientific payloads: Mars Colour Camera (MCC), Thermal Infrared Imaging Spectrometer (TIS), Methane Sensor for Mars (MSM), Lyman Alpha Photometer (LAP), and Mars Exospheric Neutral Composition Analyser (MENCA). Communications were lost in April 2022 after more than 7.5 years in orbit.""",

    "cassini_huygens_saturn": """# Cassini-Huygens Mission to Saturn
Cassini-Huygens was a joint NASA/ESA/ASI robotic spacecraft mission launched on October 15, 1997, to study Saturn, its ring system, and its natural satellites. It entered orbit around Saturn on July 1, 2004.
The ESA-built Huygens probe separated from the Cassini orbiter on December 25, 2004, and landed on Titan, Saturn's largest moon, on January 14, 2005. Huygens became the first human-made probe to land in the outer Solar System.
Cassini discovered geysers of water ice and organic molecules erupting from the south polar fractures of Enceladus, confirming a subsurface liquid ocean. On September 15, 2017, Cassini concluded its 13-year orbital mission with a planned destructive dive into Saturn's atmosphere ('The Grand Finale') to avoid potential biological contamination of Enceladus and Titan.""",

    "new_horizons_pluto": """# New Horizons Pluto and Kuiper Belt Mission
New Horizons is an interplanetary space probe launched on January 19, 2006, by NASA aboard an Atlas V rocket. It reached the highest launch speed of any human-made object at the time, leaving Earth at 16.26 km/s (58,536 km/h).
On July 14, 2015, New Horizons flew 12,500 km (7,800 mi) above the surface of Pluto, making it the first spacecraft to explore the dwarf planet and its moons (Charon, Styx, Nix, Kerberos, and Hydra).
It revealed Pluto's prominent heart-shaped nitrogen-ice plain, Tombaugh Regio, and towering water-ice mountains. On January 1, 2019, New Horizons flew past the Kuiper Belt object 486958 Arrokoth (initially nicknamed Ultima Thule), providing the first close-up images of a pristine contact binary planetesimal.""",

    "international_space_station": """# International Space Station (ISS)
The International Space Station is a modular space station in low Earth orbit. It is a multinational collaborative project involving five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada).
The first module, Zarya, was launched by a Russian Proton rocket on November 20, 1998. The station has been continuously occupied since November 2, 2000, starting with Expedition 1 (William Shepherd, Sergei Krikalev, and Yuri Gidzenko).
The ISS orbits at an average altitude of approximately 420 kilometers (260 mi) with an orbital inclination of 51.6 degrees, completing approximately 15.5 orbits per day at a speed of 7.66 km/s (27,600 km/h). The station spans 109 meters from end to end, equivalent to an American football field, and has a pressurized volume of 916 cubic meters.""",

    "artemis_program_overview": """# NASA Artemis Program
The Artemis program is a NASA-led robotic and human exploration initiative aiming to return humans to the Moon, specifically the lunar south pole region, by the mid-2020s.
The program utilizes the Space Launch System (SLS) super heavy-lift rocket and the Orion spacecraft, alongside commercial Human Landing Systems (such as SpaceX Starship HLS) and the planned lunar orbital Gateway station.
Artemis 1 was an uncrewed flight test launched on November 16, 2022, sending Orion on a 25.5-day journey around the Moon. Artemis 2 is a planned crewed lunar flyby with four astronauts (Reid Wiseman, Victor Glover, Christina Koch, and Jeremy Hansen). Artemis 3 aims to land the first woman and first person of color on the lunar surface.""",

    "europa_clipper_mission": """# Europa Clipper Mission
Europa Clipper is a NASA planetary science mission designed to study Jupiter's icy moon Europa through a series of close flybys. Europa is of high astrobiological interest because evidence strongly indicates a saltwater ocean beneath its icy crust with more than twice the volume of all Earth's oceans combined.
Europa Clipper launched on October 14, 2024, on a SpaceX Falcon Heavy rocket. It is scheduled to arrive in the Jupiter system in April 2030 after gravitational assists from Mars (February 2025) and Earth (December 2026).
The spacecraft carries nine scientific instruments, including the REASON ice-penetrating radar, the MASPEX gas spectrometer, and the Europa Imaging System (EIS). With an unfolded solar array wingspan of over 30.5 meters (100 ft), Europa Clipper is the largest spacecraft NASA has ever developed for a planetary mission.""",

    "parker_solar_probe": """# Parker Solar Probe
The Parker Solar Probe is a NASA robotic spacecraft launched on August 12, 2018, to study the outer corona of the Sun. It was named in honor of astrophysicist Eugene Parker, who first theorized the existence of the solar wind in 1958.
The spacecraft is designed to plunge through the Sun's corona, coming within 6.2 million kilometers (3.83 million miles) of the solar surface at speeds reaching 690,000 km/h (430,000 mph)—making it the fastest human-made object in history.
To survive temperatures up to 1,370 degrees Celsius (2,500 degrees Fahrenheit), the probe is protected by a 4.5-inch-thick (11.4 cm) carbon-composite heat shield coated with white ceramic paint. On April 28, 2021, the probe officially 'touched the Sun' by flying through the Alfven critical surface in the solar corona.""",

    "kepler_space_telescope": """# Kepler Space Telescope and Exoplanet Discovery
The Kepler Space Telescope was a NASA space observatory launched on March 7, 2009, to survey a portion of the Milky Way galaxy for Earth-sized exoplanets in or near habitable zones.
Kepler utilized the transit method, monitoring the brightness of more than 150,000 main-sequence stars simultaneously to detect periodic dips caused by transiting planets.
Over its primary and extended K2 mission, Kepler discovered over 2,600 confirmed exoplanets, proving that planets are more numerous than stars in our galaxy. Key discoveries included Kepler-22b (first planet confirmed in a habitable zone around a Sun-like star) and Kepler-186f (first Earth-sized planet discovered in the habitable zone of an M-dwarf star). The telescope was officially retired on October 30, 2018, after running out of onboard attitude-control fuel.""",

    "tess_space_telescope": """# Transiting Exoplanet Survey Satellite (TESS)
TESS is an all-sky exoplanet-hunting space telescope developed by MIT and NASA, launched on April 18, 2018, aboard a SpaceX Falcon 9 rocket.
Unlike Kepler, which stared continuously at a single narrow field of view, TESS monitors 85% of the sky by dividing it into 26 observation sectors, focusing on nearby bright stars located within 30 to 300 light-years of Earth.
TESS uses four wide-field optical CCD cameras with a combined field of view of 24 by 96 degrees. By 2024, TESS had identified over 7,000 candidate exoplanets (TESS Objects of Interest, or TOIs) and hundreds of confirmed exoplanets, including habitable-zone worlds like TOI-700 d and TOI-700 e.""",

    "spitzer_space_telescope": """# Spitzer Space Telescope
The Spitzer Space Telescope was an infrared space observatory launched by NASA on August 25, 2003, aboard a Delta II rocket. It was the final mission of NASA's Great Observatories program (alongside Hubble, Compton Gamma Ray Observatory, and Chandra).
Spitzer operated in an Earth-trailing heliocentric orbit, which reduced heat contamination from Earth. Its primary cryogenic phase lasted until May 15, 2009, when its liquid helium coolant was exhausted. The 'Spitzer Warm Mission' continued using two short-wavelength infrared channels until January 30, 2020.
Spitzer discovered the largest ring around Saturn (the Phoebe ring) and was pivotal in measuring the atmospheric composition and weather of exoplanets, including the discovery of seven Earth-sized planets around the TRAPPIST-1 ultracool dwarf star.""",

    "chandra_xray_observatory": """# Chandra X-ray Observatory
The Chandra X-ray Observatory is a NASA flagship space telescope launched on July 23, 1999, by Space Shuttle Columbia (STS-93). Named after Indian-American Nobel laureate Subrahmanyan Chandrasekhar, the observatory is designed to detect X-ray emissions from extremely hot and energetic regions of the Universe.
Chandra operates in a highly elliptical geocentric orbit, extending more than 130,000 kilometers away from Earth at apogee to avoid Earth's radiation belts.
Its high-resolution mirror assembly consists of four pairs of nested, grazing-incidence paraboloid and hyperboloid mirrors coated with iridium. Chandra's observations have provided fundamental evidence for dark matter in colliding galaxy clusters (such as the Bullet Cluster) and detailed studies of supermassive black holes.""",

    "rosetta_philae_mission": """# Rosetta and Philae Comet 67P Mission
Rosetta was an ESA space probe launched on March 2, 2004, to rendezvous with and orbit comet 67P/Churyumov-Gerasimenko. After a 10-year cruise through the Solar System, Rosetta entered orbit around comet 67P in August 2014.
On November 12, 2014, Rosetta deployed the Philae lander, which achieved the first soft touchdown on a comet nucleus. Philae's anchoring harpoons failed to fire, causing it to bounce twice before coming to rest in the shadow of a cliff on the comet.
Rosetta analyzed the comet's coma using the ROSINA mass spectrometer, discovering molecular oxygen (O2), noble gases, and the amino acid glycine. The mission concluded on September 30, 2016, with a controlled descent onto the comet's surface.""",

    "osiris_rex_bennu": """# OSIRIS-REx Asteroid Sample Return
OSIRIS-REx (Origins, Spectral Interpretation, Resource Identification, Security, Regolith Explorer) was a NASA asteroid study and sample-return mission launched on September 8, 2016.
The spacecraft arrived at near-Earth carbonaceous asteroid 101955 Bennu in December 2018. On October 20, 2020, OSIRIS-REx performed its Touch-And-Go (TAG) sample collection maneuver, gathering surface material using its TAGSAM arm.
On September 24, 2023, the sample capsule successfully landed at the Utah Test and Training Range, delivering 121.6 grams (4.29 oz) of pristine asteroid regolith—exceeding the mission goal of 60 grams. Analysis confirmed abundant carbon, hydrated clay minerals, and amino acid precursors. The main spacecraft was subsequently redirected to asteroid Apophis under the new name OSIRIS-APEX.""",

    "hayabusa2_ryugu": """# JAXA Hayabusa2 Asteroid Mission
Hayabusa2 was an asteroid sample-return mission operated by the Japanese space agency JAXA. Launched on December 3, 2014, aboard an H-IIA rocket, the spacecraft rendezvoused with near-Earth asteroid 162173 Ryugu in June 2018.
Hayabusa2 deployed multiple rovers (MINERVA-II1A, MINERVA-II1B, and MASCOT) and executed two touchdown sampling maneuvers, including one using a kinetic tantalum projectile to expose sub-surface material.
On December 5, 2020, Hayabusa2 returned a capsule containing 5.4 grams of Ryugu samples to Woomera, Australia. Laboratory examination revealed uracil (one of the four nucleobases in RNA), vitamin B3 (niacin), and 20 different amino acids, supporting the hypothesis that prebiotic compounds were delivered to early Earth via carbonaceous asteroids.""",

    "dart_mission": """# DART Planetary Defense Mission
The Double Asteroid Redirection Test (DART) was a NASA planetary defense mission designed to test the kinetic impactor technique for deflecting a hazardous asteroid.
Launched on November 24, 2021, DART targeted Dimorphos, a 160-meter-wide asteroid moonlet orbiting the larger asteroid Didymos. On September 26, 2022, the 570-kg DART spacecraft intentionally impacted Dimorphos at a speed of approximately 6.6 km/s (24,000 km/h).
Telescope observations confirmed that the impact shortened Dimorphos's 11-hour and 55-minute orbital period by 32 minutes—far exceeding NASA's minimum benchmark of 73 seconds. This was humanity's first successful demonstration of asteroid deflection technology. The Italian LICIACube cubesat trailed DART to photograph the impact plume.""",

    "voyager_golden_record": """# The Voyager Golden Record
The Voyager Golden Record is a phonograph record that was included aboard both Voyager 1 and Voyager 2 spacecraft launched in 1977. The records contain sounds and images selected to portray the diversity of life and culture on Earth, intended for any intelligent extraterrestrial life form that might find them.
The committee that chose the contents was chaired by Carl Sagan of Cornell University. The collection includes 115 analog images, spoken greetings in 55 languages (starting with Akkadian and ending with Wu Chinese), 90 minutes of musical selections (including Bach, Beethoven, Chuck Berry's 'Johnny B. Goode', and traditional folk music), and natural sounds such as thunder, whales, and wind.
The records are made of gold-plated copper and enclosed in an aluminum protective cover with symbolic instructions on playback speed and hydrogen transition timing.""",

    "pioneer_10_and_11": """# Pioneer 10 and Pioneer 11 Missions
Pioneer 10 and Pioneer 11 were NASA space probes launched in March 1972 and April 1973, respectively, to conduct initial reconnaissance of the asteroid belt and outer planets.
Pioneer 10 was the first spacecraft to traverse the asteroid belt and make direct observations of Jupiter (closest approach December 4, 1973). It was the first human-made object to achieve escape velocity from the Solar System.
Pioneer 11 flew by Jupiter in December 1974, using a gravitational assist to become the first spacecraft to visit Saturn in September 1979, discovering Saturn's narrow F-ring. Both probes carried the Pioneer Plaque, an engraved gold-anodized aluminum plate designed by Carl Sagan and Frank Drake featuring line drawings of a human male and female, the Sun's position relative to 14 pulsars, and the Solar System.""",

    "venera_program_venus": """# Soviet Venera Program (Venus Exploration)
The Venera series of space probes was developed by the Soviet Union between 1961 and 1984 to gather data from Venus. The program achieved several monumental space milestones.
On December 15, 1970, Venera 7 became the first spacecraft to make a successful soft landing on another planet and transmit telemetry data back to Earth, measuring a surface temperature of 475 degrees Celsius and a pressure of 90 atmospheres.
On October 22, 1975, Venera 9 landed and transmitted the first black-and-white panoramic photograph of the surface of another planet. Venera 13, which landed on March 1, 1982, transmitted the first color panoramic images of Venus's rocky terrain and recorded ambient acoustic sounds.""",

    "mariner_4_mars": """# Mariner 4 Mars Flyby
Mariner 4 was the fourth in a series of spacecraft used for planetary exploration in the flyby mode, launched by NASA on November 28, 1964, aboard an Atlas-Agena D rocket.
On July 14 and 15, 1965, Mariner 4 performed the first successful flyby of Mars, passing within 9,846 kilometers (6,118 mi) of the Martian surface. It returned 22 close-up television pictures covering about 1% of Mars's surface.
The images revealed a barren, crater-pocked terrain reminiscent of the Moon, shattering popular 19th-century speculation that Mars possessed artificial canals or abundant vegetation. Mariner 4 also measured Mars's surface atmospheric pressure to be between 4.1 and 7.0 millibars, proving the atmosphere was far thinner than previously estimated.""",

    "mariner_9_mars_orbiter": """# Mariner 9: First Mars Orbiter
Mariner 9 was a robotic NASA space probe launched on May 30, 1971, aboard an Atlas-Centaur rocket. When it entered orbit around Mars on November 14, 1971, it became the first spacecraft to orbit another planet.
Upon arrival, a massive planet-encircling dust storm obscured the Martian surface. Mission controllers waited until the storm cleared in early 1972 before initiating systematic photography.
Mariner 9 mapped 85% of Mars's surface with 7,329 images. Its major discoveries included the massive 4,000-km-long canyon system named Valles Marineris (in honor of the spacecraft), the enormous volcano Olympus Mons (the tallest known volcano in the Solar System at 21.9 km height), and dried-up river channels indicating past liquid water.""",

    "viking_1_and_2_mars": """# Viking 1 and Viking 2 Missions
NASA's Viking project consisted of two identical spacecraft, Viking 1 and Viking 2, each composed of an orbiter and a lander. Launched in August and September 1975, they arrived at Mars in the summer of 1976.
On July 20, 1976, Viking 1 Lander touched down at Chryse Planitia, becoming the first American spacecraft to land safely on Mars and operate successfully. Viking 2 Lander touched down at Utopia Planitia on September 3, 1976.
Each lander carried three biological experiments to test for metabolic activity: Pyrolytic Release, Labeled Release, and Gas Exchange. While the Labeled Release experiment gave a positive signal, the Gas Chromatograph-Mass Spectrometer (GCMS) found no organic compounds, leading scientists to attribute the reaction to non-biological soil oxidants (such as perchlorates).""",

    "mars_pathfinder_sojourner": """# Mars Pathfinder and Sojourner Rover
Mars Pathfinder was an American robotic spacecraft launched on December 4, 1996, that landed a base station with a lightweight rover on Mars's Ares Vallis on July 4, 1997.
The mission utilized an innovative direct atmospheric entry with parachutes and a cluster of 24 giant airbags to cushion touchdown.
The mission carried Sojourner, the first wheeled robotic rover to operate on Mars. Sojourner weighed just 10.5 kg (23 lb) and had six independently driven aluminum wheels on a rocker-bogie suspension system. Powered by a solar panel generating 16 Watts, Sojourner traveled approximately 100 meters over 83 sols, analyzing rocks named 'Barnacle Bill' and 'Yogi' using its APXS spectrometer.""",

    "spirit_and_opportunity_rovers": """# Mars Exploration Rovers: Spirit and Opportunity
The Mars Exploration Rover (MER) mission was a twin robotic rover program launched by NASA in June and July 2003. Spirit (MER-A) landed in Gusev Crater on January 4, 2004, and Opportunity (MER-B) landed in Meridiani Planum on January 25, 2004.
Both rovers were designed for a 90-sol primary mission but vastly exceeded their design lifespans. Spirit operated for 6 years until getting stuck in soft sand in 2010.
Opportunity operated for over 14 years until a catastrophic global dust storm cut off its solar power in June 2018. Opportunity drove a total distance of 45.16 km (28.06 miles), setting the record for the longest distance driven by any off-Earth vehicle. Opportunity discovered hematite spherules ('blueberries'), proving that acidic liquid water once soaked the Martian surface.""",

    "lucy_trojan_asteroids": """# Lucy Mission to the Trojan Asteroids
Lucy is a NASA space probe launched on October 16, 2021, aboard an Atlas V rocket. It is the first space mission designed to explore the Jupiter Trojan asteroids—swarms of primitive planetesimals trapped in Jupiter's L4 and L5 Lagrange points.
Lucy's 12-year primary mission includes flybys of two main-belt asteroids (Dinkinesh in November 2023 and Donaldjohanson in April 2025) and eight Trojan asteroids (Eurybates, Polymele, Leucus, Orus, Patroclus, and Menoetius, plus discovered satellites).
During its flyby of asteroid Dinkinesh on November 1, 2023, Lucy discovered that Dinkinesh's moonlet Selam is a contact binary asteroid. Lucy is powered by two circular solar arrays measuring 7.3 meters in diameter each.""",

    "psyche_metal_asteroid": """# NASA Psyche Asteroid Mission
Psyche is a NASA robotic orbiter mission launched on October 13, 2023, on a SpaceX Falcon Heavy rocket. The mission's target is 16 Psyche, a unique metal-rich M-type asteroid orbiting in the main asteroid belt between Mars and Jupiter.
Scientists hypothesize that 16 Psyche may be the exposed iron-nickel core of an early protoplanet that lost its rocky mantle through violent collisions during the formation of the Solar System.
The spacecraft is powered by Hall-effect electric propulsion thrusters utilizing xenon gas. It is equipped with a multispectral imager, gamma-ray and neutron spectrometer, and a magnetometer. Psyche will arrive at asteroid 16 Psyche in August 2029 and spend 26 months orbiting at decreasing altitudes.""",

    "bepicolombo_mercury": """# BepiColombo Joint Mercury Mission
BepiColombo is a joint mission between the European Space Agency (ESA) and the Japan Aerospace Exploration Agency (JAXA) to explore the planet Mercury. Launched on October 20, 2018, aboard an Ariane 5 rocket, it is the third mission to visit Mercury (after Mariner 10 and MESSENGER).
The spacecraft carries two independent orbiters stacked together: ESA's Mercury Planetary Orbiter (MPO) and JAXA's Mercury Magnetospheric Orbiter (Mio).
To brake into Mercury's steep gravitational well, BepiColombo utilizes a solar electric propulsion module and performs nine planetary gravity assists (one Earth, two Venus, and six Mercury flybys) before entering dual Mercury orbit in late 2025/early 2026.""",

    "messenger_mercury_mission": """# MESSENGER Mission to Mercury
MESSENGER (MErcury Surface, Space ENvironment, GEochemistry, and Ranging) was a NASA robotic spacecraft launched on August 3, 2004. On March 18, 2011, it became the first spacecraft to orbit Mercury.
During its four-year orbital mission, MESSENGER mapped 100% of Mercury's surface. Among its key scientific achievements, MESSENGER confirmed the presence of abundant water ice and organic compounds trapped in permanently shadowed craters at Mercury's poles.
It also discovered that Mercury possesses an unusually large metallic iron core (comprising 85% of the planet's radius) and surface volcanic plains that flooded vast regions. On April 30, 2015, MESSENGER ran out of propellant and was deorbited into Mercury's surface at 14,000 km/h.""",

    "juno_mission_jupiter": """# Juno Mission to Jupiter
Juno is a NASA New Frontiers space probe launched on August 5, 2011, aboard an Atlas V rocket. It entered a polar orbit around Jupiter on July 5, 2016.
Juno's primary scientific goals are to understand Jupiter's origin and evolution, determine the amount of water in Jupiter's deep atmosphere, measure its gravitational and magnetic fields, and investigate its massive magnetosphere.
Juno is the first mission to Jupiter powered entirely by solar panels rather than RTGs, utilizing three enormous 9-meter solar arrays. Juno discovered that Jupiter possesses a dilute, 'fuzzy' core rather than a dense, solid rocky core, and that its atmospheric jet streams extend thousands of kilometers deep into the planet.""",

    "dawn_vesta_and_ceres": """# Dawn Mission: Vesta and Ceres
NASA's Dawn mission was a robotic space probe launched on September 27, 2007, to study two of the three protoplanets of the asteroid belt: Vesta and Ceres.
Dawn used efficient xenon ion propulsion thrusters, which allowed it to become the first spacecraft to orbit two different extraterrestrial bodies.
Dawn orbited the rocky asteroid Vesta from July 2011 to September 2012, discovering a massive impact basin (Rheasilvia) with a central peak twice the height of Mount Everest. Dawn then traveled to dwarf planet Ceres, orbiting it from March 2015 until October 2018. At Ceres, Dawn discovered prominent bright sodium carbonate salt deposits in Occator Crater (Cerealia Facula) and found that Ceres is a water-rich dwarf planet with an ice-rich crust.""",

    "voyager_interstellar_findings": """# Voyager Interstellar Boundary Discoveries
Data from Voyager 1 and Voyager 2 provided the first in-situ measurements of the outer boundary of the heliosphere.
Voyager 1 detected a sharp drop in solar wind particles and a dramatic surge in galactic cosmic rays on August 25, 2012, marking its passage across the heliopause at 121.6 AU.
Voyager 2 crossed the heliopause at 119.7 AU on November 5, 2018. Unlike Voyager 1, Voyager 2's working Plasma Science (PLS) instrument allowed direct measurement of the speed, density, and temperature of interstellar plasma, showing that the interstellar medium is hotter and more dynamic than anticipated. Both spacecraft continue to transmit telemetry from beyond the solar wind.""",

    "mars_sample_return": """# Mars Sample Return Architecture
Mars Sample Return (MSR) is a proposed joint NASA and ESA multi-spacecraft campaign to return scientific samples collected by the Perseverance rover from Mars to Earth for laboratory analysis.
The baseline architecture includes three main phases:
1. Sample collection and caching by the Perseverance rover in Jezero Crater.
2. A Sample Retrieval Lander (SRL) carrying the Mars Ascent Vehicle (MAV) rocket to launch the sealed sample container into Mars orbit.
3. The Earth Return Orbiter (ERO), developed by ESA, to rendezvous with the sample container in Martian orbit and fly it back to Earth.
Return capsules are designed to safely land in a designated military testing range to undergo ultra-high containment astrobiological testing.""",

    "ingenuity_mars_helicopter": """# Ingenuity Mars Helicopter
Ingenuity was a coaxial robotic helicopter that operated on Mars between February 2021 and January 2024. Delivered to Mars attached to the belly of the Perseverance rover, Ingenuity demonstrated the first powered, controlled aerodynamic flight on another planet.
Due to Mars's thin atmosphere (less than 1% of Earth's atmospheric pressure), Ingenuity used counter-rotating carbon-fiber blades spinning at approximately 2,400 to 2,700 RPM. The helicopter weighed 1.8 kg (4.0 lb) and was powered by six lithium-ion batteries charged by a small solar panel.
Originally planned for a 30-day technology demonstration of up to five flights, Ingenuity completed 72 flights over nearly three years, flying a total distance of 17.0 km (10.6 mi) and logging 128.8 minutes of airtime before a hard landing on Flight 72 damaged its rotor blades.""",

    "envisat_earth_observation": """# Envisat Earth Observation Satellite
Envisat ('Environmental Satellite') was an advanced Earth observation satellite launched by the European Space Agency (ESA) on March 1, 2002, aboard an Ariane 5 rocket.
Operating in a sun-synchronous polar orbit at an altitude of 790 km, Envisat was the largest civilian Earth observation spacecraft ever built, weighing 8,211 kg (18,102 lb).
It carried 10 scientific instruments, including ASAR (Advanced Synthetic Aperture Radar), MERIS (Medium Resolution Imaging Spectrometer), and SCIAMACHY (atmospheric spectrometer). Envisat monitored global land cover, sea surface temperatures, greenhouse gas distributions, and ozone hole dynamics until contact was abruptly lost on April 8, 2012.""",

    "hubble_deep_field": """# Hubble Deep Field and Cosmic Expansion
The Hubble Deep Field (HDF) was a landmark astronomical observation conducted by the Hubble Space Telescope in December 1995. Over 10 consecutive days and 342 individual exposures, Hubble pointed at a tiny, seemingly empty speck of sky in the constellation Ursa Major.
The resulting composite image revealed nearly 3,000 previously unseen distant galaxies stretching back across billions of light-years of cosmic time.
Follow-up observations with the Hubble Ultra-Deep Field (2004) and eXtreme Deep Field (2012) captured galaxies formed less than 500 million years after the Big Bang. Hubble's precise measurements of Cepheid variable stars also played a key role in measuring the Hubble constant and discovering the accelerating expansion of the Universe in 1998.""",

    "exomars_trace_gas_orbiter": """# ExoMars Trace Gas Orbiter (TGO)
The ExoMars Trace Gas Orbiter is an atmospheric research spacecraft launched on March 14, 2016, by ESA and Roscosmos on a Proton-M rocket. It entered Mars orbit in October 2016.
TGO's scientific goal is to detect and map trace atmospheric gases present in concentrations below 1%, with a particular focus on methane (CH4), which could signify active biological or geological processes.
Its primary instruments include the NOMAD and ACS spectrometers, which can detect gases down to parts-per-trillion levels, and the FREND neutron detector, which maps subsurface hydrogen (water ice). TGO also acts as a vital communications relay for surface landers and rovers including Perseverance and Curiosity.""",

    "curiosity_sam_instrument": """# Sample Analysis at Mars (SAM) Instrument
The Sample Analysis at Mars (SAM) instrument suite is the heaviest and most sophisticated laboratory payload on NASA's Curiosity rover, occupying roughly half of the rover's main body.
SAM comprises three analytical instruments: a Quadrupole Mass Spectrometer (QMS), a Gas Chromatograph (GC), and a Tunable Laser Spectrometer (TLS).
SAM can heat powdered rock and soil samples up to 1,100 degrees Celsius inside tiny pyrolysis ovens to release trapped volatile compounds for chemical and isotopic analysis. SAM confirmed the presence of ancient organic molecules (such as thiophenes, benzene, and alkyl chains) preserved in 3.5-billion-year-old lacustrine mudstones at Yellowknife Bay in Gale Crater.""",

    "moxie_oxygen_generator": """# MOXIE: Mars Oxygen In-Situ Resource Utilization Experiment
MOXIE is a technology demonstration experiment on NASA's Perseverance rover that successfully proved the capability of extracting breathable oxygen from Mars's carbon dioxide-rich atmosphere.
Developed by the Massachusetts Institute of Technology (MIT), MOXIE uses solid oxide electrolysis (SOXE) to ingest Martian atmospheric CO2, filter and compress it, and heat it to 800 degrees Celsius. The electrochemical reaction splits carbon dioxide (CO2) into carbon monoxide (CO) and oxygen ions (O2).
Between April 2021 and August 2023, MOXIE completed 16 production runs, producing a total of 122 grams of oxygen with 98% purity, demonstrating a critical technology for future human missions to Mars.""",

    "tianwen_1_and_zhurong": """# Tianwen-1 and Zhurong Mars Mission
Tianwen-1 was China's first independent interplanetary mission to Mars, developed by the China National Space Administration (CNSA). Launched on July 23, 2020, aboard a Long March 5 rocket, the mission consisted of an orbiter, lander, and the Zhurong rover.
The spacecraft entered Mars orbit on February 10, 2021. On May 14, 2021, the lander touched down at Utopia Planitia. The Zhurong rover was deployed on May 22, 2021.
This made China the second nation (after the United States) to successfully land and operate a robotic rover on Mars. Zhurong traveled 1,921 meters across the Martian surface before entering planned hibernation in May 2022 due to winter sandstorms.""",

    "chandrayaan_2_orbiter": """# Chandrayaan-2 Mission and Orbiter
Chandrayaan-2 was India's second lunar exploration mission, launched on July 22, 2019, aboard a GSLV Mk III rocket from Sriharikota. The mission included an orbiter, the Vikram lander, and the Pragyan rover.
During the landing attempt on September 6, 2019, the Vikram lander deviated from its planned trajectory and crashed onto the lunar surface due to a software glitch during fine braking.
However, the Chandrayaan-2 orbiter successfully entered lunar orbit and remains fully operational. It carries eight state-of-the-art scientific payloads, including the Orbiter High Resolution Camera (OHRC), which produces the highest-resolution images (0.25 m/pixel) of the Moon from lunar orbit, and the CLASS X-ray spectrometer, which maps elemental abundance.""",

    "aditya_l1_solar_observatory": """# Aditya-L1 Solar Mission
Aditya-L1 is India's first dedicated solar observatory spacecraft, developed by ISRO. Launched on September 2, 2023, aboard a PSLV-C57 rocket from Satish Dhawan Space Centre, the spacecraft traveled 1.5 million kilometers over 126 days.
On January 6, 2024, Aditya-L1 successfully entered a halo orbit around the Sun-Earth L1 Lagrange point.
The observatory carries seven scientific payloads: four for remote sensing of the Sun (including the Visible Emission Line Coronagraph - VELC and Solar Ultraviolet Imaging Telescope - SUIT) and three for in-situ observations of solar wind particles and magnetic fields (including ASPEX, PAPA, and MAG). The mission studies coronal heating, coronal mass ejections (CMEs), and space weather dynamics.""",

    "slims_moon_sniper": """# Smart Lander for Investigating Moon (SLIM)
SLIM ('Moon Sniper') was a precision lunar lander mission developed by the Japan Aerospace Exploration Agency (JAXA). Launched on September 6, 2023, aboard an H-IIA rocket alongside the XRISM telescope, SLIM aimed to achieve pinpoint landing accuracy within 100 meters.
On January 19, 2024, SLIM touched down near Shioli crater. Telemetry confirmed it landed just 55 meters from its target, achieving the most accurate extraterrestrial landing in history.
Although SLIM rolled upside down during touchdown due to an engine thruster failure, its solar panels generated power as the sun angle shifted, allowing it to survive multiple freezing two-week lunar nights. The mission deployed two LEV (Lunar Excursion Vehicle) mini-probes.""",

    "galileo_jupiter_mission": """# Galileo Spacecraft Mission to Jupiter
Galileo was an American robotic space probe launched on October 18, 1989, aboard Space Shuttle Atlantis (STS-34). It arrived at Jupiter on December 7, 1995, becoming the first spacecraft to orbit the gas giant.
Galileo released an atmospheric entry probe that descended 150 km into Jupiter's atmosphere, measuring extreme winds of 600 km/h and intense lightning.
During its eight-year orbital mission, Galileo provided evidence of a subsurface liquid water ocean on Europa, detected a magnetic field generated by Ganymede (the only moon known to have one), and observed volcanic eruptions on Io. The mission was deliberately ended on September 21, 2003, by plunging Galileo into Jupiter's dense atmosphere to protect Europa from potential terrestrial microbes."""
}

print(f"Writing {len(DOCS)} domain documents to {DOCUMENTS_DIR}...")
for doc_name, content in DOCS.items():
    file_path = DOCUMENTS_DIR / f"{doc_name}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

print("Documents created successfully!")

# 60 Curated Evaluation Questions across 4 distinct categories
QUESTIONS = [
    # Category 1: Direct Fact Extraction (Verifiable, precise facts from corpus)
    {"id": "Q01", "category": "Direct_Fact", "question": "On what exact date did Apollo 11 land on the lunar surface, and in which lunar region?", "ground_truth": "Apollo 11 landed on July 20, 1969, in Tranquility Base (Mare Tranquillitatis).", "in_corpus": "True", "source_doc": "apollo_11_mission"},
    {"id": "Q02", "category": "Direct_Fact", "question": "What is the primary power source for the Curiosity Mars rover?", "ground_truth": "A Multi-Mission Radioisotope Thermoelectric Generator (MMRTG) using plutonium-238 dioxide.", "in_corpus": "True", "source_doc": "curiosity_rover_mars"},
    {"id": "Q03", "category": "Direct_Fact", "question": "How many total flights did the Ingenuity helicopter complete on Mars before sustaining damage?", "ground_truth": "Ingenuity completed 72 flights.", "in_corpus": "True", "source_doc": "ingenuity_mars_helicopter"},
    {"id": "Q04", "category": "Direct_Fact", "question": "Where is the James Webb Space Telescope located in space?", "ground_truth": "At the Sun-Earth L2 Lagrange point, approximately 1.5 million kilometers from Earth.", "in_corpus": "True", "source_doc": "james_webb_space_telescope"},
    {"id": "Q05", "category": "Direct_Fact", "question": "What is the diameter and material composition of the primary mirror of the James Webb Space Telescope?", "ground_truth": "6.5 meters in diameter, consisting of 18 hexagonal beryllium segments coated with gold.", "in_corpus": "True", "source_doc": "james_webb_space_telescope"},
    {"id": "Q06", "category": "Direct_Fact", "question": "Which space agency developed the Huygens probe and on which celestial body did it land?", "ground_truth": "Developed by the European Space Agency (ESA) and landed on Saturn's moon Titan on January 14, 2005.", "in_corpus": "True", "source_doc": "cassini_huygens_saturn"},
    {"id": "Q07", "category": "Direct_Fact", "question": "What total mass of asteroid sample did OSIRIS-REx deliver back to Earth from asteroid Bennu?", "ground_truth": "121.6 grams (4.29 oz).", "in_corpus": "True", "source_doc": "osiris_rex_bennu"},
    {"id": "Q08", "category": "Direct_Fact", "question": "What was the name and landing site coordinates of India's Chandrayaan-3 lander?", "ground_truth": "Vikram lander landed at Shiv Shakti Point (69.37° S, 32.35° E) near the lunar south pole.", "in_corpus": "True", "source_doc": "chandrayaan_3_mission"},
    {"id": "Q09", "category": "Direct_Fact", "question": "What was the total budget cost of India's Mars Orbiter Mission (Mangalyaan)?", "ground_truth": "Approximately $74 million USD (Rs 450 crore).", "in_corpus": "True", "source_doc": "mangalyaan_mars_orbiter"},
    {"id": "Q10", "category": "Direct_Fact", "question": "What is the primary target and propulsion type of NASA's Psyche mission?", "ground_truth": "Asteroid 16 Psyche (a metal-rich M-type asteroid) using Hall-effect electric propulsion thrusters fueled by xenon gas.", "in_corpus": "True", "source_doc": "psyche_metal_asteroid"},
    {"id": "Q11", "category": "Direct_Fact", "question": "On what date did Voyager 1 enter interstellar space?", "ground_truth": "August 25, 2012.", "in_corpus": "True", "source_doc": "voyager_1_probe"},
    {"id": "Q12", "category": "Direct_Fact", "question": "What was the name of the first wheeled robotic rover to operate on Mars?", "ground_truth": "Sojourner rover, delivered by the Mars Pathfinder mission in 1997.", "in_corpus": "True", "source_doc": "mars_pathfinder_sojourner"},
    {"id": "Q13", "category": "Direct_Fact", "question": "What key technology did MOXIE demonstrate on Mars?", "ground_truth": "Extracting breathable oxygen from Martian carbon dioxide using solid oxide electrolysis.", "in_corpus": "True", "source_doc": "moxie_oxygen_generator"},
    {"id": "Q14", "category": "Direct_Fact", "question": "How much did the DART impact shorten Dimorphos's orbital period around Didymos?", "ground_truth": "By 32 minutes (from 11 hours 55 minutes).", "in_corpus": "True", "source_doc": "dart_mission"},
    {"id": "Q15", "category": "Direct_Fact", "question": "What orbital location and launch date correspond to India's Aditya-L1 solar observatory?", "ground_truth": "Halo orbit around Sun-Earth L1 Lagrange point, launched on September 2, 2023.", "in_corpus": "True", "source_doc": "aditya_l1_solar_observatory"},
    {"id": "Q16", "category": "Direct_Fact", "question": "What is the landing accuracy achieved by Japan's SLIM 'Moon Sniper' lander?", "ground_truth": "55 meters from its target site near Shioli crater.", "in_corpus": "True", "source_doc": "slims_moon_sniper"},
    {"id": "Q17", "category": "Direct_Fact", "question": "What are the names of the two outer planets visited exclusively by Voyager 2?", "ground_truth": "Uranus and Neptune.", "in_corpus": "True", "source_doc": "voyager_2_probe"},
    {"id": "Q18", "category": "Direct_Fact", "question": "Which space shuttle mission deployed the Hubble Space Telescope in 1990?", "ground_truth": "Space Shuttle Discovery (STS-31) on April 24, 1990.", "in_corpus": "True", "source_doc": "hubble_space_telescope"},

    # Category 2: Multi-Hop Synthesis (Requires combining facts across sections or docs)
    {"id": "Q19", "category": "Multi_Hop", "question": "Compare the power sources of the Curiosity rover, Juno spacecraft, and Voyager 1 probe.", "ground_truth": "Curiosity and Voyager 1 use radioisotope thermoelectric generators (RTGs fueled by plutonium-238), whereas Juno is powered by three large solar panel arrays.", "in_corpus": "True", "source_doc": "curiosity_rover_mars, juno_mission_jupiter, voyager_1_probe"},
    {"id": "Q20", "category": "Multi_Hop", "question": "Which two spacecraft have returned physical samples from near-Earth asteroids, and how much sample mass did each return?", "ground_truth": "Hayabusa2 returned 5.4 grams from asteroid Ryugu, and OSIRIS-REx returned 121.6 grams from asteroid Bennu.", "in_corpus": "True", "source_doc": "hayabusa2_ryugu, osiris_rex_bennu"},
    {"id": "Q21", "category": "Multi_Hop", "question": "How do the observational approaches of Kepler and TESS differ in surveying the sky for exoplanets?", "ground_truth": "Kepler monitored a single fixed narrow field of stars continuously, while TESS is an all-sky survey dividing 85% of the sky into 26 sectors to monitor nearby bright stars.", "in_corpus": "True", "source_doc": "kepler_space_telescope, tess_space_telescope"},
    {"id": "Q22", "category": "Multi_Hop", "question": "What is the difference in primary mission destination and landing mechanism between Mars Pathfinder and Chandrayaan-3?", "ground_truth": "Mars Pathfinder landed on Mars (Ares Vallis) using a parachute and 24 cushioning airbags, whereas Chandrayaan-3 soft-landed on the lunar south pole using throttleable rocket engines.", "in_corpus": "True", "source_doc": "mars_pathfinder_sojourner, chandrayaan_3_mission"},
    {"id": "Q23", "category": "Multi_Hop", "question": "Which two space telescopes are positioned at the Sun-Earth L1 and L2 Lagrange points respectively?", "ground_truth": "Aditya-L1 is at the Sun-Earth L1 Lagrange point, and James Webb Space Telescope (JWST) is at the Sun-Earth L2 Lagrange point.", "in_corpus": "True", "source_doc": "aditya_l1_solar_observatory, james_webb_space_telescope"},
    {"id": "Q24", "category": "Multi_Hop", "question": "Compare the landing sites and operational lifespans of the Spirit and Opportunity rovers on Mars.", "ground_truth": "Spirit landed in Gusev Crater and operated for 6 years; Opportunity landed in Meridiani Planum and operated for over 14 years, driving 45.16 km.", "in_corpus": "True", "source_doc": "spirit_and_opportunity_rovers"},
    {"id": "Q25", "category": "Multi_Hop", "question": "Which space missions confirmed water ice on the Moon and Mercury respectively?", "ground_truth": "Chandrayaan-1 confirmed water ice on the Moon via its Moon Mineralogy Mapper, and MESSENGER confirmed water ice in permanently shadowed polar craters on Mercury.", "in_corpus": "True", "source_doc": "chandrayaan_1_mission, messenger_mercury_mission"},
    {"id": "Q26", "category": "Multi_Hop", "question": "How did the atmospheric entry strategies and outcomes differ between Galileo at Jupiter and Venera 7 at Venus?", "ground_truth": "Galileo released a probe that descended 150 km into Jupiter's gaseous atmosphere before being crushed/melted, whereas Venera 7 successfully soft-landed on Venus's solid surface and transmitted data for 23 minutes.", "in_corpus": "True", "source_doc": "galileo_jupiter_mission, venera_program_venus"},
    {"id": "Q27", "category": "Multi_Hop", "question": "What were the primary organic or prebiotic discoveries made by Rosetta on comet 67P and Hayabusa2 on asteroid Ryugu?", "ground_truth": "Rosetta discovered the amino acid glycine and molecular oxygen, while Hayabusa2 discovered uracil (RNA nucleobase), vitamin B3, and 20 amino acids.", "in_corpus": "True", "source_doc": "rosetta_philae_mission, hayabusa2_ryugu"},
    {"id": "Q28", "category": "Multi_Hop", "question": "What are the distinct scientific roles of the ChemCam instrument on Curiosity and the MOXIE experiment on Perseverance?", "ground_truth": "ChemCam uses a laser spectrometer to analyze rock and soil elemental composition, whereas MOXIE is an in-situ resource experiment that generates oxygen from carbon dioxide.", "in_corpus": "True", "source_doc": "curiosity_rover_mars, moxie_oxygen_generator"},
    {"id": "Q29", "category": "Multi_Hop", "question": "Compare the distances traveled beyond the heliopause by Voyager 1 and Voyager 2 at their respective crossing dates.", "ground_truth": "Voyager 1 crossed at 121.6 AU in August 2012, while Voyager 2 crossed at 119.7 AU in November 2018.", "in_corpus": "True", "source_doc": "voyager_1_probe, voyager_2_probe, voyager_interstellar_findings"},
    {"id": "Q30", "category": "Multi_Hop", "question": "List three robotic spacecraft that have successfully soft-landed rovers on Mars and their corresponding countries of origin.", "ground_truth": "1. Mars Pathfinder (Sojourner) - USA, 2. Mars 2020 (Perseverance) - USA, 3. Tianwen-1 (Zhurong) - China.", "in_corpus": "True", "source_doc": "mars_pathfinder_sojourner, perseverance_rover_mars, tianwen_1_and_zhurong"},

    # Category 3: Unanswerable / Out-of-Corpus (Hallucination traps - model should refuse or say 'I don't know')
    {"id": "Q31", "category": "Out_Of_Corpus", "question": "What was the exact battery capacity in kilowatt-hours of the fictional Apollo 18 landing module on Mars?", "ground_truth": "I do not have enough information in the provided context to answer this question (Apollo 18 Mars mission is fictional / not in corpus).", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q32", "category": "Out_Of_Corpus", "question": "What was the brand name of the solar panels installed on the Hermes rover during the 2019 Europa landing mission?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q33", "category": "Out_Of_Corpus", "question": "How many metric tons of titanium were extracted by the Artemis 5 mining station on the Moon in 2022?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q34", "category": "Out_Of_Corpus", "question": "What is the serial number of the laser oscillator unit installed on the Neptune Triton orbiter in 2020?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q35", "category": "Out_Of_Corpus", "question": "Which astronaut became the first person to walk on Saturn's moon Titan in June 2021?", "ground_truth": "I do not have enough information in the provided context to answer this question (no human has ever walked on Titan).", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q36", "category": "Out_Of_Corpus", "question": "What was the exact price in euros paid by ESA to purchase the Plutonian orbital map from ISRO in 2018?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q37", "category": "Out_Of_Corpus", "question": "What was the name of the captain of the Soviet submarine that recovered the Voyager 3 probe from the Arctic ocean?", "ground_truth": "I do not have enough information in the provided context to answer this question (Voyager 3 does not exist).", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q38", "category": "Out_Of_Corpus", "question": "How many passengers boarded the commercial Virgin Orbit lunar shuttle flight in November 1985?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q39", "category": "Out_Of_Corpus", "question": "What brand of espresso machine was carried inside the Huygens probe when landing on Titan?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q40", "category": "Out_Of_Corpus", "question": "Which German university constructed the primary reactor for the Chandrayaan-4 submarine probe in 2015?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q41", "category": "Out_Of_Corpus", "question": "What was the diameter of the diamond drill bit used by Mariner 4 on the surface of Mars?", "ground_truth": "I do not have enough information in the provided context to answer this question (Mariner 4 was a flyby mission without a drill or landing).", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q42", "category": "Out_Of_Corpus", "question": "What is the title of the song by Elvis Presley included on the James Webb Space Telescope optical disk?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q43", "category": "Out_Of_Corpus", "question": "How many kilograms of liquid nitrogen were delivered to the ISS by the Apollo 13 crew?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q44", "category": "Out_Of_Corpus", "question": "What was the name of the pet dog sent aboard the New Horizons spacecraft to Pluto in 2006?", "ground_truth": "I do not have enough information in the provided context to answer this question (no animal was sent aboard New Horizons).", "in_corpus": "False", "source_doc": "none"},
    {"id": "Q45", "category": "Out_Of_Corpus", "question": "What was the retail price in USD of the tickets sold to tourists for the 1976 Viking 1 Mars flight?", "ground_truth": "I do not have enough information in the provided context to answer this question.", "in_corpus": "False", "source_doc": "none"},

    # Category 4: Subtle Adversarial & Common Misconceptions (Tests precise grounding against hallucination priors)
    {"id": "Q46", "category": "Adversarial_Misconception", "question": "Did the Viking biological experiments in 1976 definitively prove the existence of living microorganisms on Mars?", "ground_truth": "No. Although the Labeled Release experiment gave a positive signal, the GCMS found no organic compounds, and scientists attributed the reaction to non-biological soil oxidants like perchlorates.", "in_corpus": "True", "source_doc": "viking_1_and_2_mars"},
    {"id": "Q47", "category": "Adversarial_Misconception", "question": "Was Neil Armstrong the only astronaut who walked on the Moon during the Apollo 11 mission?", "ground_truth": "No. Buzz Aldrin also walked on the Moon with Armstrong, spending about two hours and 15 minutes outside.", "in_corpus": "True", "source_doc": "apollo_11_mission"},
    {"id": "Q48", "category": "Adversarial_Misconception", "question": "Did the Hubble Space Telescope travel to the Moon or Mars to take its deep space photos?", "ground_truth": "No. Hubble operates in low Earth orbit around Earth, outside atmospheric distortion.", "in_corpus": "True", "source_doc": "hubble_space_telescope"},
    {"id": "Q49", "category": "Adversarial_Misconception", "question": "Was Voyager 1 the first spacecraft to visit both Uranus and Neptune?", "ground_truth": "No. Voyager 2 is the only spacecraft to have visited Uranus and Neptune. Voyager 1 only visited Jupiter and Saturn.", "in_corpus": "True", "source_doc": "voyager_1_probe, voyager_2_probe"},
    {"id": "Q50", "category": "Adversarial_Misconception", "question": "Did the Parker Solar Probe land on the solid surface of the Sun?", "ground_truth": "No. The Sun has no solid surface; the probe flew through the solar corona (Alfven critical surface) protected by a heat shield.", "in_corpus": "True", "source_doc": "parker_solar_probe"},
    {"id": "Q51", "category": "Adversarial_Misconception", "question": "Was the Chandrayaan-2 Vikram lander successful in executing a soft landing on the Moon in 2019?", "ground_truth": "No. The Vikram lander crashed during descent due to a software glitch, though the Chandrayaan-2 orbiter remained successful.", "in_corpus": "True", "source_doc": "chandrayaan_2_orbiter"},
    {"id": "Q52", "category": "Adversarial_Misconception", "question": "Did the DART spacecraft capture Dimorphos and bring it back into Earth orbit?", "ground_truth": "No. DART was a kinetic impactor that intentionally crashed into Dimorphos to alter its orbital period around Didymos.", "in_corpus": "True", "source_doc": "dart_mission"},
    {"id": "Q53", "category": "Adversarial_Misconception", "question": "Did the New Horizons mission land on the surface of Pluto to collect soil samples?", "ground_truth": "No. New Horizons was a flyby mission that flew 12,500 km above Pluto's surface without landing.", "in_corpus": "True", "source_doc": "new_horizons_pluto"},
    {"id": "Q54", "category": "Adversarial_Misconception", "question": "Is the James Webb Space Telescope an optical-only telescope that replaced Hubble in low Earth orbit?", "ground_truth": "No. JWST operates primarily in infrared astronomy and is located at the Sun-Earth L2 Lagrange point, not in low Earth orbit.", "in_corpus": "True", "source_doc": "james_webb_space_telescope"},
    {"id": "Q55", "category": "Adversarial_Misconception", "question": "Did Mariner 4 discover artificial water canals constructed by civilizations on Mars?", "ground_truth": "No. Mariner 4 showed Mars to be a cratered, barren world and shattered the 19th-century canal speculation.", "in_corpus": "True", "source_doc": "mariner_4_mars"},
    {"id": "Q56", "category": "Adversarial_Misconception", "question": "Was the Opportunity rover powered by a nuclear plutonium generator during its 14-year mission?", "ground_truth": "No. Opportunity was solar-powered; its mission ended when a 2018 global dust storm blocked solar illumination.", "in_corpus": "True", "source_doc": "spirit_and_opportunity_rovers"},
    {"id": "Q57", "category": "Adversarial_Misconception", "question": "Did the Cassini spacecraft return to Earth and land in the Pacific Ocean after exploring Saturn?", "ground_truth": "No. Cassini completed a planned destructive plunge ('Grand Finale') into Saturn's atmosphere in September 2017 to prevent contamination of its moons.", "in_corpus": "True", "source_doc": "cassini_huygens_saturn"},
    {"id": "Q58", "category": "Adversarial_Misconception", "question": "Was India the first nation in history to ever land a spacecraft on the Moon?", "ground_truth": "No. India was the fourth nation to achieve a soft lunar landing (after USSR, USA, China), but the first to land near the lunar south polar region.", "in_corpus": "True", "source_doc": "chandrayaan_3_mission"},
    {"id": "Q59", "category": "Adversarial_Misconception", "question": "Did the Philae lander fire its harpoons and stay rigidly locked to comet 67P on its initial touchdown?", "ground_truth": "No. Philae's anchoring harpoons failed to fire, causing it to bounce twice before settling in the shadow of a cliff.", "in_corpus": "True", "source_doc": "rosetta_philae_mission"},
    {"id": "Q60", "category": "Adversarial_Misconception", "question": "Is the International Space Station anchored to a geostationary orbital slot directly above Europe?", "ground_truth": "No. The ISS is in low Earth orbit at ~420 km altitude and travels at 7.66 km/s, orbiting Earth approximately 15.5 times per day.", "in_corpus": "True", "source_doc": "international_space_station"}
]

print(f"Writing {len(QUESTIONS)} benchmark questions to {QUESTIONS_FILE}...")
with open(QUESTIONS_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "category", "question", "ground_truth", "in_corpus", "source_doc"])
    writer.writeheader()
    writer.writerows(QUESTIONS)

print("Evaluation dataset created successfully!")
