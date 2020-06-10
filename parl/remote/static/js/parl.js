function createDivs(res, divs) {
  var elem = document.getElementById("main");
  var worker_num = res.workers.length;    // 8
  var curr_num = elem.children.length;    // 0
  divs = (divs < worker_num) ? divs : worker_num;

  if (curr_num < divs) {
    for (var i = curr_num; i < divs; i++) {
      var workerDiv = document.createElement("div");
      workerDiv.id = `w${i}`;
      if (i === 0) {
        workerDiv.innerHTML = `<div class="card-header" id="${i}"><div style="display:inline;">Master</div><div style="float:right; display:inlene;" id="cpu">[CPU] ${res.total_vacant_cpus}/${res.total_cpus}</div></div>`;
      } else {
        workerDiv.innerHTML = `<p class="card-header" id="${i}">Worker ${res.workers[i].hostname}</p>`;
      }

      var cardDiv = document.createElement("div");
      cardDiv.className = "row";
      var card = '';
      for (var j = 0; j < 3; j++)
        card += `<div class="col-lg-4"><div class="card mb-1"><div id="w${i}c${j}" class="card-body" style="height: 110px;"></div></div></div>`;
      cardDiv.innerHTML = card;
      workerDiv.appendChild(cardDiv);
      elem.appendChild(workerDiv);

      for (var j = 0; j < 3; j++)
        imgHandle[`w${i}c${j}`] = echarts.init(document.getElementById(`w${i}c${j}`));
    };
  } else if (curr_num > worker_num) {
    for (var i = curr_num - 1; i >= worker_num; i--) {
      delete imgHandle[`w${i}c0`];
      delete imgHandle[`w${i}c1`];
      delete imgHandle[`w${i}c2`];

      var workerDiv = document.getElementById(`w${i}`);
      elem.removeChild(workerDiv);
    }
  }
}


function addPlots(res, record, imgHandle, begin, end) {
  var worker_num = res.workers.length;
  var record_num = Object.keys(record).length;

  end = (end < worker_num) ? end : worker_num;
  for (var i = begin; i < end; i++) {
    var worker = res.workers[i];

    var cpuOption = {
      color: ["#7B68EE", "#6495ED"],
      legend: {
        orient: 'vertical',
        x: 'left',
        data: ['Used CPU', 'Vacant CPU'],
        textStyle: {
          fontSize: 8,
        }
      },
      series: [
        {
          type: "pie",
          radius: "80%",
          label: {
            normal: {
              formatter: "{c}",
              show: true,
              position: "inner",
              fontSize: 16,
            }
          },
          data: [
            { value: worker.used_cpus, name: "Used CPU" },
            { value: worker.vacant_cpus, name: "Vacant CPU" }
          ]
        }
      ]
    };

    var memoryOption = {
      color: ["#FF8C00", "#FF4500"],
      legend: {
        orient: "vertical",
        x: "left",
        data: ["Used Memory", "Vacant Memory"],
        textStyle: {
          fontSize: 8,
        }        
      },
      series: [
        {
          name: "Memory",
          type: "pie",
          radius: "80%",
          label: {
            normal: {
              formatter: "{c}",
              show: true,
              position: "inner",
              fontSize: 12,
            }
          },

          data: [
            { value: worker.used_memory, name: "Used Memory" },
            { value: worker.vacant_memory, name: "Vacant Memory" }
          ]
        }
      ]
    };

    var loadOption = {
      grid:{
        x:30,
        y:25,
        x2:20,
        y2:20,
        borderWidth:1
    },      
      xAxis: {
        type: "category",
        data: worker.load_time,
      },
      yAxis: {
        type: "value",
        name: "Average CPU load (%)",
        splitNumber:3,
        nameTextStyle:{
          padding: [0, 0, 0, 60],
          fontSize: 10,
      }        
      },
      series: [
        {
          data: worker.load_value,
          type: "line"
        }
      ]
    };

    var cpuNum = document.getElementById('cpu');
    cpu.innerText = `[CPU] ${res.total_vacant_cpus}/${res.total_cpus}`

    if (i < record_num && worker.hostname === record[i].hostname) {
      if (worker.used_cpus !== record[i].used_cpus) {
        imgHandle[`w${i}c0`].setOption(cpuOption);
      }
      if (worker.used_memory !== record[i].used_memory) {
        imgHandle[`w${i}c1`].setOption(memoryOption);
      }
      imgHandle[`w${i}c2`].setOption(loadOption);      
    } else {
      if (i > 0){
        var workerTitle = document.getElementById(`${i}`);
        workerTitle.innerText = `Worker ${worker.hostname}`
      }
      imgHandle[`w${i}c0`].setOption(cpuOption);
      imgHandle[`w${i}c1`].setOption(memoryOption);
      imgHandle[`w${i}c2`].setOption(loadOption);      
    }

    record[i] = {
      hostname: worker.hostname,
      used_cpus: worker.used_cpus,
      vacant_cpus: worker.vacant_cpus,
      used_memory: worker.used_memory,
      vacant_memory: worker.vacant_memory
    };
  }

  if (end < record_num) {
    for (var i = end; i < record_num; i++)
      delete record[i]
  }
};

function autoTable(res) {
  var table = document.getElementById("table");
  table.innerHTML = "";
  var rows = res.clients.length;
  for(var i=0; i< rows; i++){
    var tr = document.createElement('tr');
    var s1 = `<th scope="row">${i+1}</th>`;
    var s2 = `<td>${res.clients[i].file_path}</td>`;
    var s3 = `<td>${res.clients[i].client_address}</td>`;
    var s4 = `<td>${res.clients[i].actor_num}</td>`;
    var s5 = `<td>${res.clients[i].time}</td>`;
    var s6 = `<td><a href=${res.clients[i].log_monitor_url}>link</a></td>`;
    tr.innerHTML = s1 + s2 + s3 + s4 + s5 + s6;
    table.appendChild(tr);
  }
};
